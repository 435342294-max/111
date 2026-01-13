#include <boost/timer/timer.hpp>
#include <ctime>
#include <boost/filesystem.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include <set>
#include <map>
#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include <opencv2/opencv.hpp>
#include <cmath>

#include "utils/Logger.h"
#include "utils/Config.h"

#include "segment_modeling/UBlock.h"
#include "segment_modeling/UBlock_building.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <sys/resource.h>

namespace py = pybind11;
namespace fs = boost::filesystem;

// CGAL 类型定义
typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef CGAL::Point_set_3<Point_3> Point_set;
typedef CGAL::Parallel_tag Concurrency_tag;

// 坐标系结构体
struct GeoTransform {
    double gt[6]; // GDAL地理变换参数
    bool valid;
    
    GeoTransform() : valid(false) {
        for (int i = 0; i < 6; i++) gt[i] = 0.0;
    }
    
    // 像素坐标转地理坐标
    void pixelToGeo(double pixelX, double pixelY, double& geoX, double& geoY) const {
        if (!valid) {
            geoX = pixelX;
            geoY = pixelY;
            return;
        }
        geoX = gt[0] + pixelX * gt[1] + pixelY * gt[2];
        geoY = gt[3] + pixelX * gt[4] + pixelY * gt[5];
    }
    
    // 地理坐标转像素坐标
    void geoToPixel(double geoX, double geoY, double& pixelX, double& pixelY) const {
        if (!valid) {
            pixelX = geoX;
            pixelY = geoY;
            return;
        }
        // 逆变换公式
        double det = gt[1] * gt[5] - gt[2] * gt[4];
        if (fabs(det) < 1e-10) {
            pixelX = geoX;
            pixelY = geoY;
            return;
        }
        pixelX = (gt[5] * (geoX - gt[0]) - gt[2] * (geoY - gt[3])) / det;
        pixelY = (-gt[4] * (geoX - gt[0]) + gt[1] * (geoY - gt[3])) / det;
    }
    
    // 获取图像的地理范围
    void getImageBounds(int width, int height, 
                       double& minX, double& maxX, 
                       double& minY, double& maxY) const {
        if (!valid) {
            minX = 0; maxX = width;
            minY = 0; maxY = height;
            return;
        }
        
        double cornersX[4], cornersY[4];
        
        // 四个角点
        pixelToGeo(0, 0, cornersX[0], cornersY[0]);  // 左上
        pixelToGeo(width, 0, cornersX[1], cornersY[1]); // 右上
        pixelToGeo(width, height, cornersX[2], cornersY[2]); // 右下
        pixelToGeo(0, height, cornersX[3], cornersY[3]); // 左下
        
        minX = std::min({cornersX[0], cornersX[1], cornersX[2], cornersX[3]});
        maxX = std::max({cornersX[0], cornersX[1], cornersX[2], cornersX[3]});
        minY = std::min({cornersY[0], cornersY[1], cornersY[2], cornersY[3]});
        maxY = std::max({cornersY[0], cornersY[1], cornersY[2], cornersY[3]});
    }
    
    // 计算像素坐标在另一个GeoTransform中的对应位置
    bool transformToOther(const GeoTransform& other, 
                         double srcPixelX, double srcPixelY,
                         double& dstPixelX, double& dstPixelY) const {
        if (!valid || !other.valid) return false;
        
        double geoX;
        double geoY;
        pixelToGeo(srcPixelX, srcPixelY, geoX, geoY);
        other.geoToPixel(geoX, geoY, dstPixelX, dstPixelY);
        return true;
    }
};

void copyBandToChannel(cv::Mat& dest, const cv::Mat& src, int channel) {
    if (dest.empty() || src.empty()) {
        LOG_ERROR << "Empty matrices in copyBandToChannel";
        return;
    }
    
    if (dest.rows != src.rows || dest.cols != src.cols) {
        LOG_ERROR << "Size mismatch in copyBandToChannel";
        return;
    }
    
    if (channel >= dest.channels()) {
        LOG_ERROR << "Channel index out of bounds: " << channel << " >= " << dest.channels();
        return;
    }
    
    // 确保通道类型匹配
    if (src.type() != CV_8U && src.type() != CV_16U) {
        LOG_ERROR << "Unsupported source type: " << src.type();
        return;
    }
    
    // 使用更安全的方法复制通道数据
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            if (src.type() == CV_8U) {
                dest.at<cv::Vec3b>(y, x)[channel] = src.at<uchar>(y, x);
            } else if (src.type() == CV_16U) {
                dest.at<cv::Vec3w>(y, x)[channel] = src.at<ushort>(y, x);
            }
        }
    }
}

/**
 * @brief 检测DSM的单位并返回转换因子
 */
double detect_dsm_unit_factor(const cv::Mat& dsm_img) {
    if (dsm_img.type() == CV_16U) {
        // 16位DSM，分析典型值范围
        double min_val, max_val;
        cv::minMaxLoc(dsm_img, &min_val, &max_val);
        
        LOG_INFO << "DSM raw value range: " << min_val << " to " << max_val;
        
        // 如果最大值在1000以内，可能是米单位
        if (max_val < 1000.0) {
            LOG_INFO << "DSM appears to be in meters (no conversion needed)";
            return 1.0;
        }
        // 如果最大值在10000以内，可能是分米单位（*0.1）
        else if (max_val < 10000.0) {
            LOG_INFO << "DSM appears to be in decimeters, converting to meters (factor 0.1)";
            return 0.1;
        }
        // 如果最大值在100000以内，可能是厘米单位（*0.01）
        else if (max_val < 100000.0) {
            LOG_INFO << "DSM appears to be in centimeters, converting to meters (factor 0.01)";
            return 0.01;
        }
        // 否则可能是毫米单位（*0.001）
        else {
            LOG_INFO << "DSM appears to be in millimeters, converting to meters (factor 0.001)";
            return 0.001;
        }
    }
    else if (dsm_img.type() == CV_32F) {
        // 32位浮点，直接使用
        double min_val, max_val;
        cv::minMaxLoc(dsm_img, &min_val, &max_val);
        LOG_INFO << "DSM float range: " << min_val << " to " << max_val;
        
        if (max_val > 1000.0) {
            LOG_WARNING << "Float DSM values unusually high, may need conversion";
            // 如果大于1000，尝试转换为米
            if (max_val > 10000.0) {
                return 0.01;  // 厘米转米
            }
        }
        
        return 1.0;
    }
    
    return 1.0;
}

// 读取TIFF文件并获取地理信息
bool readTIFFWithGeoInfo(const std::string& filepath, cv::Mat& image, GeoTransform& geoTransform) {
    GDALDataset* dataset = (GDALDataset*)GDALOpen(filepath.c_str(), GA_ReadOnly);
    if (!dataset) {
        LOG_ERROR << "Failed to open TIFF file: " << filepath;
        return false;
    }
    
    int width = dataset->GetRasterXSize();
    int height = dataset->GetRasterYSize();
    int bands = dataset->GetRasterCount();
    
    // 获取地理变换参数
    if (dataset->GetGeoTransform(geoTransform.gt) == CE_None) {
        geoTransform.valid = true;
        LOG_DEBUG << "GeoTransform for " << filepath << ": "
                 << geoTransform.gt[0] << ", " << geoTransform.gt[1] << ", " << geoTransform.gt[2] << ", "
                 << geoTransform.gt[3] << ", " << geoTransform.gt[4] << ", " << geoTransform.gt[5];
    } else {
        geoTransform.valid = false;
        LOG_WARNING << "No GeoTransform found for " << filepath;
    }
    
    LOG_INFO << "Reading " << filepath << ": " << width << "x" << height << " with " << bands << " bands";
    
    // 读取图像数据
    if (bands == 1) {
        // 单波段图像
        image.create(height, width, CV_16U);
        CPLErr err = dataset->GetRasterBand(1)->RasterIO(
            GF_Read, 0, 0, width, height, 
            image.data, width, height, GDT_UInt16, 0, 0);
        if (err != CE_None) {
            LOG_ERROR << "Failed to read raster data from: " << filepath;
            GDALClose(dataset);
            return false;
        }
        LOG_INFO << "Loaded single-band image (CV_16U)";
    } 
    else if (bands == 3 || bands == 4) {
        // 多波段图像（RGB或RGBA）
        // 首先获取第一个波段的数据类型
        GDALRasterBand* band = dataset->GetRasterBand(1);
        GDALDataType dataType = band->GetRasterDataType();
        
        LOG_INFO << "Band data type: " << GDALGetDataTypeName(dataType);
        
        if (dataType == GDT_Byte) {
            // 8位图像
            if (bands == 3) {
                image.create(height, width, CV_8UC3);
                LOG_INFO << "Creating CV_8UC3 image";
            } else if (bands == 4) {
                image.create(height, width, CV_8UC4);
                LOG_INFO << "Creating CV_8UC4 image";
            }
            
            // 正确读取每个波段
            for (int b = 0; b < bands; b++) {
                cv::Mat band_img(height, width, CV_8U);
                band = dataset->GetRasterBand(b + 1);
                CPLErr err = band->RasterIO(
                    GF_Read, 0, 0, width, height,
                    band_img.data, width, height, GDT_Byte, 0, 0);
                if (err != CE_None) {
                    LOG_ERROR << "Failed to read band " << b << " from: " << filepath;
                    GDALClose(dataset);
                    return false;
                }
                
                // 确保图像有正确的通道数
                if (image.channels() >= b + 1) {
                    // 将单通道复制到多通道图像的正确位置
                    cv::Mat channel = image.colRange(b, b + 1).reshape(1, height);
                    band_img.copyTo(channel);
                }
            }
        } 
        else if (dataType == GDT_UInt16) {
            // 16位图像
            if (bands == 3) {
                image.create(height, width, CV_16UC3);
                LOG_INFO << "Creating CV_16UC3 image";
            } else if (bands == 4) {
                image.create(height, width, CV_16UC4);
                LOG_INFO << "Creating CV_16UC4 image";
            }
            
            for (int b = 0; b < bands; b++) {
                cv::Mat band_img(height, width, CV_16U);
                band = dataset->GetRasterBand(b + 1);
                CPLErr err = band->RasterIO(
                    GF_Read, 0, 0, width, height,
                    band_img.data, width, height, GDT_UInt16, 0, 0);
                if (err != CE_None) {
                    LOG_ERROR << "Failed to read band " << b << " from: " << filepath;
                    GDALClose(dataset);
                    return false;
                }
                
                if (image.channels() >= b + 1) {
                    cv::Mat channel = image.colRange(b, b + 1).reshape(1, height);
                    band_img.copyTo(channel);
                }
            }
        }
        else {
            LOG_ERROR << "Unsupported data type for multi-band image: " << GDALGetDataTypeName(dataType);
            GDALClose(dataset);
            return false;
        }
    }
    else {
        LOG_ERROR << "Unsupported number of bands: " << bands;
        GDALClose(dataset);
        return false;
    }
    
    GDALClose(dataset);
    LOG_INFO << "Successfully read image: " << image.cols << "x" << image.rows 
             << " channels: " << image.channels() << " type: " << image.type();
    return true;
}

/**
 * @brief 改进版本：考虑分辨率差异的坐标转换，从大范围分割图中提取与DSM相同地理范围的区域
 */
bool readSegmentationRegion_resolution_aware(const std::string& seg_path, 
                                           const GeoTransform& dsm_gt, int dsm_width, int dsm_height,
                                           cv::Mat& seg_region, GeoTransform& seg_gt) {
    GDALDataset* seg_dataset = (GDALDataset*)GDALOpen(seg_path.c_str(), GA_ReadOnly);
    if (!seg_dataset) {
        LOG_ERROR << "Failed to open segmentation file: " << seg_path;
        return false;
    }
    
    // 获取分割图的地理信息
    if (seg_dataset->GetGeoTransform(seg_gt.gt) == CE_None) {
        seg_gt.valid = true;
        LOG_DEBUG << "Segmentation GeoTransform: "
                 << seg_gt.gt[0] << ", " << seg_gt.gt[1] << ", " << seg_gt.gt[2] << ", "
                 << seg_gt.gt[3] << ", " << seg_gt.gt[4] << ", " << seg_gt.gt[5];
    } else {
        seg_gt.valid = false;
        LOG_WARNING << "No GeoTransform found for segmentation";
        GDALClose(seg_dataset);
        return false;
    }
    
    int seg_width = seg_dataset->GetRasterXSize();
    int seg_height = seg_dataset->GetRasterYSize();
    
    LOG_INFO << "Segmentation image size: " << seg_width << "x" << seg_height;
    
    // 计算DSM的地理范围
    double dsm_minX, dsm_maxX, dsm_minY, dsm_maxY;
    dsm_gt.getImageBounds(dsm_width, dsm_height, dsm_minX, dsm_maxX, dsm_minY, dsm_maxY);
    
    LOG_INFO << "DSM bounds:";
    LOG_INFO << "  X range: [" << dsm_minX << ", " << dsm_maxX << "]";
    LOG_INFO << "  Y range: [" << dsm_minY << ", " << dsm_maxY << "]";
    LOG_INFO << "  Pixel size X: " << dsm_gt.gt[1] << " degrees/pixel";
    LOG_INFO << "  Pixel size Y: " << dsm_gt.gt[5] << " degrees/pixel";
    
    // 将DSM的地理范围转换为分割图的像素坐标
    double seg_pixel_left, seg_pixel_top, seg_pixel_right, seg_pixel_bottom;
    
    // 左上角 (minX, maxY)
    seg_gt.geoToPixel(dsm_minX, dsm_maxY, seg_pixel_left, seg_pixel_top);
    // 右下角 (maxX, minY)
    seg_gt.geoToPixel(dsm_maxX, dsm_minY, seg_pixel_right, seg_pixel_bottom);
    
    // 确保坐标顺序正确
    if (seg_pixel_left > seg_pixel_right) std::swap(seg_pixel_left, seg_pixel_right);
    if (seg_pixel_top > seg_pixel_bottom) std::swap(seg_pixel_top, seg_pixel_bottom);
    
    LOG_DEBUG << "Segmentation pixel coordinates before rounding:";
    LOG_DEBUG << "  Left/Top: (" << seg_pixel_left << ", " << seg_pixel_top << ")";
    LOG_DEBUG << "  Right/Bottom: (" << seg_pixel_right << ", " << seg_pixel_bottom << ")";
    
    // 转换为整数像素坐标（使用四舍五入）
    int xoff = static_cast<int>(std::round(seg_pixel_left));
    int yoff = static_cast<int>(std::round(seg_pixel_top));
    int xsize = static_cast<int>(std::round(seg_pixel_right - seg_pixel_left));
    int ysize = static_cast<int>(std::round(seg_pixel_bottom - seg_pixel_top));
    
    // 确保在图像范围内
    xoff = std::max(0, xoff);
    yoff = std::max(0, yoff);
    xsize = std::min(xsize, seg_width - xoff);
    ysize = std::min(ysize, seg_height - yoff);
    
    if (xsize <= 0 || ysize <= 0) {
        LOG_ERROR << "Invalid region size after coordinate conversion: " 
                 << xsize << "x" << ysize;
        
        // 调试：打印更多信息
        LOG_ERROR << "DSM bounds: X=[" << dsm_minX << "," << dsm_maxX 
                  << "], Y=[" << dsm_minY << "," << dsm_maxY << "]";
        
        // 检查分割图的地理范围
        double seg_minX, seg_maxX, seg_minY, seg_maxY;
        seg_gt.getImageBounds(seg_width, seg_height, seg_minX, seg_maxX, seg_minY, seg_maxY);
        LOG_ERROR << "Segmentation bounds: X=[" << seg_minX << "," << seg_maxX 
                  << "], Y=[" << seg_minY << "," << seg_maxY << "]";
        
        GDALClose(seg_dataset);
        return false;
    }
    
    LOG_INFO << "Reading segmentation region: offset=(" << xoff << "," << yoff 
             << "), size=" << xsize << "x" << ysize;
    
    // 读取区域数据
    seg_region.create(ysize, xsize, CV_8U);
    CPLErr err = seg_dataset->GetRasterBand(1)->RasterIO(
        GF_Read, xoff, yoff, xsize, ysize,
        seg_region.data, xsize, ysize, GDT_Byte, 0, 0);
    
    GDALClose(seg_dataset);
    
    if (err != CE_None) {
        LOG_ERROR << "Failed to read segmentation region";
        return false;
    }
    
    LOG_INFO << "Successfully read segmentation region: " 
             << seg_region.cols << "x" << seg_region.rows;
    
    // 如果分割图区域尺寸与DSM不同，需要重采样
    if (seg_region.size() != cv::Size(dsm_width, dsm_height)) {
        double scale_x = static_cast<double>(dsm_width) / seg_region.cols;
        double scale_y = static_cast<double>(dsm_height) / seg_region.rows;
        
        LOG_INFO << "Resampling required: segmentation " << seg_region.cols << "x" << seg_region.rows
                 << " to DSM " << dsm_width << "x" << dsm_height;
        LOG_INFO << "Scale factors: X=" << scale_x << ", Y=" << scale_y;
        
        cv::Mat resized;
        cv::resize(seg_region, resized, cv::Size(dsm_width, dsm_height), 
                  0, 0, cv::INTER_NEAREST);
        seg_region = resized;
        
        LOG_INFO << "Resampled segmentation to " << seg_region.cols << "x" << seg_region.rows;
    }
    
    return true;
}

void log_memory_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    LOG_INFO << "Memory usage: " << usage.ru_maxrss << " KB";
}

// 可视化法线图
cv::Mat visualize_normal_map(const cv::Mat& normal_map) {
    cv::Mat visual_normal_map;
    cv::Mat normalized_normal_map = normal_map + cv::Scalar(1.0, 1.0, 1.0);
    normalized_normal_map *= 127.5;
    normalized_normal_map.convertTo(visual_normal_map, CV_8UC3);
    return visual_normal_map;
}

// 保存法线图到文件
void save_normal_map(const cv::Mat& normal_map, const std::string& file_path) {
    cv::Mat visual_normal_map = visualize_normal_map(normal_map);
    if (!cv::imwrite(file_path, visual_normal_map)) {
        std::cerr << "Error: Could not save the normal map to " << file_path << std::endl;
    }
}

/**
 * @brief 稳健的局部平面拟合算法计算法线
 */
cv::Mat compute_normal_map_robust(const cv::Mat& dsm_img, int window_size = 3) {
    cv::Mat normal_map(dsm_img.size(), CV_64FC3, cv::Scalar(0.0, 0.0, 1.0));
    
    cv::Mat dsm_double;
    dsm_img.convertTo(dsm_double, CV_64F);
    
    int half_win = window_size / 2;
    
    #pragma omp parallel for
    for (int y = half_win; y < dsm_double.rows - half_win; ++y) {
        for (int x = half_win; x < dsm_double.cols - half_win; ++x) {
            std::vector<cv::Point3d> points;
            double center_z = dsm_double.at<double>(y, x);
            
            for (int dy = -half_win; dy <= half_win; ++dy) {
                for (int dx = -half_win; dx <= half_win; ++dx) {
                    double z = dsm_double.at<double>(y + dy, x + dx);
                    if (std::abs(z - center_z) < 5.0) {
                        points.push_back(cv::Point3d(dx, dy, z));
                    }
                }
            }
            
            if (points.size() < 3) {
                normal_map.at<cv::Vec3d>(y, x) = cv::Vec3d(0.0, 0.0, 1.0);
                continue;
            }
            
            double sum_x = 0, sum_y = 0, sum_z = 0;
            for (const auto& p : points) {
                sum_x += p.x;
                sum_y += p.y;
                sum_z += p.z;
            }
            
            int n = points.size();
            double mean_x = sum_x / n;
            double mean_y = sum_y / n;
            double mean_z = sum_z / n;
            
            cv::Mat A(n, 3, CV_64F);
            cv::Mat B(n, 1, CV_64F);
            
            for (int i = 0; i < n; ++i) {
                A.at<double>(i, 0) = points[i].x - mean_x;
                A.at<double>(i, 1) = points[i].y - mean_y;
                A.at<double>(i, 2) = 1.0;
                B.at<double>(i, 0) = points[i].z - mean_z;
            }
            
            cv::Mat X;
            cv::solve(A, B, X, cv::DECOMP_SVD);
            
            double a = X.at<double>(0, 0);
            double b = X.at<double>(1, 0);
            
            cv::Vec3d normal(-a, -b, 1.0);
            double length = cv::norm(normal);
            
            if (length > 1e-6) {
                normal /= length;
            }
            
            normal_map.at<cv::Vec3d>(y, x) = normal;
        }
    }
    
    return normal_map;
}

/**
 * @brief 针对卫星数据优化的DSINE处理（优化版）
 * 移除坐标变换和临时文件IO，直接处理图像空间法线
 */
cv::Mat compute_normal_map_dsine_satellite_optimized(const cv::Mat& tdom_img, 
                                                   const std::string& model_path = "",
                                                   const std::string& device = "cuda") {
    try {
        py::module sys = py::module::import("sys");
        
        std::string dsine_path = "/home/hzt/Structured_Reconstruction/code_lod/DSINE-main/";
        if (fs::exists(dsine_path)) {
            sys.attr("path").attr("append")(dsine_path);
        }
        
        py::module dsine_module;
        try {
            dsine_module = py::module::import("dsine.inference");
        } catch (const py::error_already_set& e) {
            LOG_WARNING << "Cannot import dsine.inference, trying alternative import...";
            dsine_module = py::module::import("inference");
        }
        
        // 处理四通道TDOM图像
        cv::Mat rgb_input;
        if (tdom_img.channels() == 4) {
            // 四通道RGBA转三通道RGB
            cv::cvtColor(tdom_img, rgb_input, cv::COLOR_BGRA2RGB);
        } else if (tdom_img.channels() == 3) {
            // BGR转RGB
            cv::cvtColor(tdom_img, rgb_input, cv::COLOR_BGR2RGB);
        } else if (tdom_img.channels() == 1) {
            // 灰度转RGB
            cv::cvtColor(tdom_img, rgb_input, cv::COLOR_GRAY2RGB);
        } else {
            LOG_ERROR << "Unsupported number of channels: " << tdom_img.channels();
            cv::Mat default_normal(tdom_img.size(), CV_64FC3, cv::Scalar(0.0, 0.0, 1.0));
            return default_normal;
        }
        
        // 调整图像尺寸为DSINE输入要求（384x384）
        cv::Mat resized_input;
        int target_size = 384;
        cv::resize(rgb_input, resized_input, cv::Size(target_size, target_size));
        
        // 直接传递numpy数组给DSINE，避免文件IO
        py::array img_array = py::array_t<uint8_t>(
            {resized_input.rows, resized_input.cols, 3},
            resized_input.data
        );
        
        py::object result;
        if (!model_path.empty()) {
            result = dsine_module.attr("infer")(img_array, model_path, device);
        } else {
            result = dsine_module.attr("infer")(img_array, device);
        }
        
        py::array_t<float> normal_np = result.cast<py::array_t<float>>();
        auto buf = normal_np.request();
        
        int height = buf.shape[0];
        int width = buf.shape[1];
        
        // DSINE输出的是相机坐标系下的法线图（归一化）
        cv::Mat normal_camera(height, width, CV_32FC3, buf.ptr);
        
        // 调整到原始尺寸
        cv::Mat normal_resized;
        cv::resize(normal_camera, normal_resized, rgb_input.size(), 0, 0, cv::INTER_LINEAR);
        
        // 转换为双精度
        cv::Mat normal_double;
        normal_resized.convertTo(normal_double, CV_64FC3);
        
        // 对于卫星数据，DSINE输出的法线图已经是在图像空间中的
        // 我们只需要确保法线指向正确的方向（对于卫星数据，Z应该为正，即指向天空）
        cv::Mat normal_world(normal_double.size(), CV_64FC3);
        
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < normal_double.rows; ++y) {
            for (int x = 0; x < normal_double.cols; ++x) {
                cv::Vec3d n_img = normal_double.at<cv::Vec3d>(y, x);
                
                // 确保法线指向正确的方向（对于卫星数据，Z应该为正）
                // DSINE输出的是相机坐标系，对于俯视的卫星图像，我们需要调整Z方向
                if (n_img[2] < 0) {
                    n_img = -n_img;
                }
                
                // 归一化
                double length = cv::norm(n_img);
                if (length > 1e-6) {
                    n_img /= length;
                } else {
                    n_img = cv::Vec3d(0.0, 0.0, 1.0);
                }
                
                normal_world.at<cv::Vec3d>(y, x) = n_img;
            }
        }
        
        return normal_world;
        
    } catch (const py::error_already_set& e) {
        LOG_ERROR << "Python error in DSINE satellite: " << e.what();
        cv::Mat default_normal(tdom_img.size(), CV_64FC3, cv::Scalar(0.0, 0.0, 1.0));
        return default_normal;
    } catch (const std::exception& e) {
        LOG_ERROR << "Error in compute_normal_map_dsine_satellite: " << e.what();
        cv::Mat default_normal(tdom_img.size(), CV_64FC3, cv::Scalar(0.0, 0.0, 1.0));
        return default_normal;
    }
}

/**
 * @brief 基于像素的法向引导加权平滑
 * 使用法线相似性和空间距离的加权平均，避免块状伪影
 */
cv::Mat smooth_dsm_with_normals_pixelwise(const cv::Mat& dsm, const cv::Mat& normals, 
                                        const cv::Mat& mask, 
                                        int window_radius = 5,
                                        double spatial_sigma = 2.0,
                                        double normal_sigma = 0.2) {
    cv::Mat smoothed_dsm = dsm.clone();
    
    int width = dsm.cols;
    int height = dsm.rows;
    
    // 预计算空间权重核
    int kernel_size = 2 * window_radius + 1;
    cv::Mat spatial_weights(kernel_size, kernel_size, CV_64F);
    
    for (int dy = -window_radius; dy <= window_radius; ++dy) {
        for (int dx = -window_radius; dx <= window_radius; ++dx) {
            double dist_sq = dx*dx + dy*dy;
            double weight = exp(-dist_sq / (2.0 * spatial_sigma * spatial_sigma));
            spatial_weights.at<double>(dy + window_radius, dx + window_radius) = weight;
        }
    }
    
    #pragma omp parallel for
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // 只在建筑区域内进行平滑
            if (mask.at<uchar>(y, x) == 0) continue;
            
            cv::Vec3d center_normal = normals.at<cv::Vec3d>(y, x);
            double center_height = dsm.at<double>(y, x);
            
            double total_weight = 0.0;
            double weighted_sum = 0.0;
            
            // 遍历窗口内的像素
            for (int dy = -window_radius; dy <= window_radius; ++dy) {
                for (int dx = -window_radius; dx <= window_radius; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    // 边界检查
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                    
                    // 只在建筑区域内考虑
                    if (mask.at<uchar>(ny, nx) == 0) continue;
                    
                    // 空间权重
                    double spatial_weight = spatial_weights.at<double>(dy + window_radius, 
                                                                       dx + window_radius);
                    
                    // 法线相似性权重
                    cv::Vec3d neighbor_normal = normals.at<cv::Vec3d>(ny, nx);
                    double normal_similarity = std::abs(center_normal.dot(neighbor_normal));
                    double normal_weight = exp(-(1.0 - normal_similarity) / (2.0 * normal_sigma * normal_sigma));
                    
                    // 高度差异权重（防止异常值影响）
                    double neighbor_height = dsm.at<double>(ny, nx);
                    double height_diff = std::abs(neighbor_height - center_height);
                    double height_weight = exp(-height_diff / 5.0); // 5米阈值
                    
                    // 总权重 = 空间权重 × 法线权重 × 高度权重
                    double total_weight_pixel = spatial_weight * normal_weight * height_weight;
                    
                    weighted_sum += total_weight_pixel * neighbor_height;
                    total_weight += total_weight_pixel;
                }
            }
            
            // 应用加权平均
            if (total_weight > 1e-6) {
                smoothed_dsm.at<double>(y, x) = weighted_sum / total_weight;
            }
        }
    }
    
    return smoothed_dsm;
}

/**
 * @brief 改进的建筑掩码提取函数，适应黑白二值分割图
 */
cv::Mat extract_building_mask_from_binary(const cv::Mat& segmentation_img) {
    cv::Mat building_mask;
    
    if (segmentation_img.channels() == 1) {
        // 单波段图像，假设是黑白二值图
        // 对于黑白二值图，建筑通常是白色（255）或接近白色
        // 但也要考虑可能有中间值
        
        // 首先统计像素值分布
        std::map<int, int> value_counts;
        for (int y = 0; y < segmentation_img.rows; ++y) {
            const uchar* ptr = segmentation_img.ptr<uchar>(y);
            for (int x = 0; x < segmentation_img.cols; ++x) {
                value_counts[ptr[x]]++;
            }
        }
        
        LOG_INFO << "Segmentation value distribution (top 10):";
        int count = 0;
        for (const auto& [value, freq] : value_counts) {
            if (count++ < 10) {
                LOG_INFO << "  Value " << static_cast<int>(value) << ": " << freq << " pixels";
            }
        }
        
        // 自动确定阈值：寻找明显的分界点
        // 对于黑白二值图，通常会有大量的0（背景）和255（建筑）
        int max_freq_value = 0;
        int max_freq = 0;
        for (const auto& [value, freq] : value_counts) {
            if (freq > max_freq && value < 250) { // 排除可能的噪声
                max_freq = freq;
                max_freq_value = value;
            }
        }
        
        // 假设建筑是像素值较高的区域
        int threshold = 128; // 默认阈值
        if (value_counts.find(255) != value_counts.end() && value_counts[255] > 100) {
            // 如果有大量255像素，使用255作为建筑
            threshold = 254;
        } else if (value_counts.find(1) != value_counts.end() && value_counts[1] > 100) {
            // 如果有大量1像素，使用1作为建筑
            threshold = 0;
        }
        
        LOG_INFO << "Using threshold: " << threshold << " for building extraction";
        
        // 应用阈值
        if (threshold == 0) {
            // 建筑值为1，背景为0
            cv::threshold(segmentation_img, building_mask, 0, 255, cv::THRESH_BINARY);
        } else {
            // 建筑值为高值（如255）
            cv::threshold(segmentation_img, building_mask, threshold, 255, cv::THRESH_BINARY);
        }
        
        // 添加形态学操作填充小的空洞和去除噪声
        cv::Mat kernel_small = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(building_mask, building_mask, cv::MORPH_CLOSE, kernel_small);
        cv::morphologyEx(building_mask, building_mask, cv::MORPH_OPEN, kernel_small);
        
        // 去除太小的连通区域（可能是噪声）
        std::vector<std::vector<cv::Point>> small_contours;
        cv::findContours(building_mask.clone(), small_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat cleaned_mask = cv::Mat::zeros(building_mask.size(), CV_8U);
        
        for (size_t i = 0; i < small_contours.size(); i++) {
            double area = cv::contourArea(small_contours[i]);
            if (area > 50) { // 保留面积大于50像素的区域
                cv::drawContours(cleaned_mask, small_contours, i, cv::Scalar(255), cv::FILLED);
            }
        }
        
        building_mask = cleaned_mask;
        
    } else {
        LOG_ERROR << "Unsupported segmentation image format: " << segmentation_img.channels() << " channels";
        building_mask = cv::Mat::zeros(segmentation_img.size(), CV_8U);
    }
    
    int building_pixels = cv::countNonZero(building_mask);
    LOG_INFO << "Building mask extracted: " << building_pixels << " pixels";
    
    return building_mask;
}

/**
 * @brief 处理卫星数据，生成建筑三维模型（改进版，处理不同分辨率）
 */

bool process_building_contour_enhanced(const cv::Mat& smoothed_dsm, 
                                     const cv::Mat& tdom_img,
                                     const cv::Mat& global_normal_map,
                                     const cv::Mat& building_mask,
                                     const GeoTransform& dsm_gt,
                                     double target_resolution,
                                     int contour_index,
                                     const std::vector<cv::Point>& contour,
                                     int& building_index,
                                     std::vector<std::unique_ptr<UBlock_building>>& buildings,
                                     std::map<int, std::pair<cv::Rect, double>>& building_info,
                                     const std::string& output_dir) {
    
    try {
        // 1. 检查轮廓是否有效
        if (contour.empty()) {
            LOG_WARNING << "Contour " << contour_index << " is empty";
            return false;
        }
        
        // 2. 检查轮廓面积
        double area = cv::contourArea(contour) * (target_resolution * target_resolution);
        if (area < 100.0) { // 最小面积
            LOG_DEBUG << "Skipping building " << contour_index << " with area " << area << " m²";
            return false;
        }
        
        // 3. 检查轮廓点数
        if (contour.size() < 3) {
            LOG_WARNING << "Contour " << contour_index << " has too few points: " << contour.size();
            return false;
        }
        
        // 4. 创建单个建筑掩码
        cv::Mat building_mask_individual = cv::Mat::zeros(building_mask.size(), CV_8U);
        cv::drawContours(building_mask_individual, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(255), cv::FILLED);
        
        // 5. 获取边界框并检查
        cv::Rect bounding_rect = cv::boundingRect(contour);
        
        // 检查边界框是否有效
        if (bounding_rect.width <= 0 || bounding_rect.height <= 0) {
            LOG_WARNING << "Invalid bounding box for building " << contour_index;
            return false;
        }
        
        // 检查边界框是否在图像范围内
        if (bounding_rect.x < 0 || bounding_rect.y < 0 || 
            bounding_rect.x + bounding_rect.width > smoothed_dsm.cols ||
            bounding_rect.y + bounding_rect.height > smoothed_dsm.rows) {
            LOG_WARNING << "Bounding box out of bounds for building " << contour_index;
            return false;
        }
        
        int padding = 5; // 减少padding以避免越界
        bounding_rect.x = std::max(0, bounding_rect.x - padding);
        bounding_rect.y = std::max(0, bounding_rect.y - padding);
        bounding_rect.width = std::min(smoothed_dsm.cols - bounding_rect.x, 
                                      bounding_rect.width + 2 * padding);
        bounding_rect.height = std::min(smoothed_dsm.rows - bounding_rect.y, 
                                       bounding_rect.height + 2 * padding);
        
        // 再次检查边界框
        if (bounding_rect.width <= 0 || bounding_rect.height <= 0) {
            LOG_WARNING << "Invalid bounding box after padding for building " << contour_index;
            return false;
        }
        
        // 6. 提取裁剪区域
        cv::Mat cropped_dsm, cropped_tdom, cropped_mask, cropped_normal;
        
        try {
            cropped_dsm = smoothed_dsm(bounding_rect).clone();
            cropped_tdom = tdom_img(bounding_rect).clone();
            cropped_mask = building_mask_individual(bounding_rect).clone();
            cropped_normal = global_normal_map(bounding_rect).clone();
        } catch (const cv::Exception& e) {
            LOG_ERROR << "Failed to crop regions for building " << contour_index << ": " << e.what();
            return false;
        }
        
        // 7. 检查裁剪区域是否为空
        if (cropped_dsm.empty() || cropped_tdom.empty() || cropped_mask.empty() || cropped_normal.empty()) {
            LOG_WARNING << "One or more cropped regions are empty for building " << contour_index;
            return false;
        }
        
        // 8. 检查裁剪区域尺寸是否一致
        if (cropped_dsm.size() != cropped_mask.size() ||
            cropped_tdom.size() != cropped_mask.size() ||
            cropped_normal.size() != cropped_mask.size()) {
            LOG_WARNING << "Cropped region size mismatch for building " << contour_index;
            return false;
        }
        
        // 9. 创建掩码后的数据
        cv::Mat masked_dsm = cv::Mat::zeros(cropped_dsm.size(), cropped_dsm.type());
        cv::Mat masked_tdom = cv::Mat::zeros(cropped_tdom.size(), cropped_tdom.type());
        cv::Mat masked_normal = cv::Mat::zeros(cropped_normal.size(), cropped_normal.type());
        
        cropped_dsm.copyTo(masked_dsm, cropped_mask);
        cropped_tdom.copyTo(masked_tdom, cropped_mask);
        cropped_normal.copyTo(masked_normal, cropped_mask);
        
        // 10. 检查掩码后数据是否有效
        if (cv::countNonZero(cropped_mask) == 0) {
            LOG_WARNING << "No building pixels in mask for building " << contour_index;
            return false;
        }
        
        // 11. 计算建筑高度
        std::vector<double> building_heights;
        double building_min_height = std::numeric_limits<double>::max();
        double building_max_height = std::numeric_limits<double>::lowest();
        
        for (int y = 0; y < masked_dsm.rows; ++y) {
            const uchar* mask_ptr = cropped_mask.ptr<uchar>(y);
            const double* dsm_ptr = masked_dsm.ptr<double>(y);
            
            for (int x = 0; x < masked_dsm.cols; ++x) {
                if (mask_ptr[x] > 0) {
                    double height = dsm_ptr[x];
                    // 过滤异常高度值
                    if (height > 1000.0 || height < -100.0) {
                        continue;
                    }
                    building_heights.push_back(height);
                    building_min_height = std::min(building_min_height, height);
                    building_max_height = std::max(building_max_height, height);
                }
            }
        }
        
        // 12. 检查是否有有效的高度数据
        if (building_heights.empty()) {
            LOG_WARNING << "No valid height data for building " << contour_index;
            return false;
        }
        
        // 13. 使用统计方法处理高度
        std::sort(building_heights.begin(), building_heights.end());
        double median_height = building_heights[building_heights.size() / 2];
        double q1 = building_heights[building_heights.size() / 4];
        double q3 = building_heights[building_heights.size() * 3 / 4];
        double iqr = q3 - q1;
        
        // 使用IQR方法去除异常值
        double lower_bound = q1 - 1.5 * iqr;
        double upper_bound = q3 + 1.5 * iqr;
        
        // 重新计算高度范围
        building_min_height = median_height - iqr * 0.5;
        building_max_height = median_height + iqr * 0.5;
        
        double building_height = building_max_height - building_min_height;
        
        // 14. 检查最终建筑高度
        if (building_height <= 0 || building_height > 500.0) {
            LOG_WARNING << "Invalid building height for building " << contour_index << ": " << building_height << "m";
            return false;
        }
        
        // 15. 创建位置图
        cv::Mat position_map(masked_dsm.size(), CV_64FC3);
        
        for (int y = 0; y < position_map.rows; ++y) {
            cv::Vec3d* pos_ptr = position_map.ptr<cv::Vec3d>(y);
            const double* dsm_ptr = masked_dsm.ptr<double>(y);
            const uchar* mask_ptr = cropped_mask.ptr<uchar>(y);
            
            for (int x = 0; x < position_map.cols; ++x) {
                double geo_x, geo_y;
                int global_pixel_x = bounding_rect.x + x;
                int global_pixel_y = bounding_rect.y + y;
                
                if (dsm_gt.valid) {
                    dsm_gt.pixelToGeo(global_pixel_x, global_pixel_y, geo_x, geo_y);
                } else {
                    geo_x = global_pixel_x * target_resolution;
                    geo_y = global_pixel_y * target_resolution;
                }
                
                double height = building_min_height;
                if (mask_ptr[x] > 0) {
                    height = std::max(building_min_height, std::min(dsm_ptr[x], building_max_height));
                }
                
                pos_ptr[x] = cv::Vec3d(geo_x, geo_y, height - building_min_height);
            }
        }
        
        // 16. 检查位置图是否有效
        if (position_map.empty()) {
            LOG_WARNING << "Position map creation failed for building " << contour_index;
            return false;
        }
        
        // 17. 创建建筑对象
        std::string building_name = output_dir + "/building_" + std::to_string(building_index);
        
        LOG_INFO << "Creating building object " << building_index 
                 << " with size: " << building_mask_individual.cols << "x" << building_mask_individual.rows;
        
        // 确保所有输入矩阵不为空
        if (building_mask_individual.empty() || masked_tdom.empty() || 
            position_map.empty() || masked_normal.empty()) {
            LOG_ERROR << "One or more input matrices are empty for building " << contour_index;
            return false;
        }
        
        // 检查尺寸一致性
        if (building_mask_individual.size() != masked_tdom.size() ||
            building_mask_individual.size() != position_map.size() ||
            building_mask_individual.size() != masked_normal.size()) {
            LOG_ERROR << "Size mismatch for building " << contour_index;
            return false;
        }
        
        // 尝试创建建筑对象
        try {
            auto building_ptr = std::make_unique<UBlock_building>(
                building_name,
                building_mask_individual,
                masked_tdom,
                position_map,
                masked_normal,
                building_min_height,
                building_max_height,
                Mesh_3()
            );
            
            // 如果成功，添加到列表
            buildings.push_back(std::move(building_ptr));
            
            // 保存建筑信息
            building_info[building_index] = std::make_pair(bounding_rect, building_min_height);
            
            LOG_INFO << "Building " << building_index 
                     << ": area=" << std::fixed << std::setprecision(1) << area << " m², "
                     << "height=" << building_height << " m, "
                     << "bbox=[" << bounding_rect.x << "," << bounding_rect.y << "] "
                     << bounding_rect.width << "x" << bounding_rect.height;
            
            building_index++;
            return true;
            
        } catch (const std::exception& e) {
            LOG_ERROR << "Failed to create UBlock_building object for building " << contour_index << ": " << e.what();
            return false;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR << "Unexpected error processing building " << contour_index << ": " << e.what();
        return false;
    }
}

/**
 * @brief 处理卫星数据，生成建筑三维模型（安全增强版）
 */
void process_satellite_data_safe(const std::string& dsm_path,
                                const std::string& tdom_path,
                                const std::string& segmentation_path,
                                const std::string& output_dir,
                                std::vector<std::unique_ptr<UBlock_building>>& buildings,
                                bool use_dsine = false,
                                const std::string& dsine_model_path = "",
                                double target_resolution = 0.5,
                                double min_building_area = 100.0) {
    
    if (!fs::exists(dsm_path) || !fs::exists(tdom_path) || !fs::exists(segmentation_path)) {
        throw std::runtime_error("One or more input files do not exist.");
    }
    
    LOG_INFO << "Starting satellite data processing...";
    LOG_INFO << "DSM: " << dsm_path;
    LOG_INFO << "TDOM: " << tdom_path;
    LOG_INFO << "Segmentation: " << segmentation_path;
    
    // 1. 读取DSM和TDOM
    cv::Mat dsm_img;
    GeoTransform dsm_gt;
    if (!readTIFFWithGeoInfo(dsm_path, dsm_img, dsm_gt)) {
        throw std::runtime_error("Failed to read DSM TIFF file.");
    }
    
    cv::Mat tdom_img;
    GeoTransform tdom_gt;
    if (!readTIFFWithGeoInfo(tdom_path, tdom_img, tdom_gt)) {
        throw std::runtime_error("Failed to read TDOM TIFF file.");
    }
    
    LOG_INFO << "DSM size: " << dsm_img.cols << "x" << dsm_img.rows;
    LOG_INFO << "TDOM size: " << tdom_img.cols << "x" << tdom_img.rows;
    
    // 2. 检查尺寸一致性
    if (dsm_img.size() != tdom_img.size()) {
        LOG_WARNING << "DSM and TDOM have different sizes, resizing TDOM to match DSM";
        cv::resize(tdom_img, tdom_img, dsm_img.size(), 0, 0, cv::INTER_LINEAR);
    }
    
    // 3. 读取分割图
    cv::Mat segmentation_img;
    GeoTransform seg_gt;
    if (!readSegmentationRegion_resolution_aware(segmentation_path, dsm_gt, 
                                               dsm_img.cols, dsm_img.rows,
                                               segmentation_img, seg_gt)) {
        throw std::runtime_error("Failed to read segmentation region.");
    }
    
    LOG_INFO << "Segmentation region size: " << segmentation_img.cols << "x" << segmentation_img.rows;
    
    // 4. 处理DSM图像
    double dsm_conversion_factor = detect_dsm_unit_factor(dsm_img);
    LOG_INFO << "Using DSM conversion factor: " << dsm_conversion_factor;
    
    cv::Mat dsm_float;
    if (dsm_img.type() == CV_16U) {
        dsm_img.convertTo(dsm_float, CV_64F, dsm_conversion_factor);
    } else {
        dsm_img.convertTo(dsm_float, CV_64F);
    }
    
    // 5. 从分割图像中提取建筑掩码
    cv::Mat building_mask = extract_building_mask_from_binary(segmentation_img);
    LOG_INFO << "Building mask created: " << cv::countNonZero(building_mask) << " pixels";
    
    // 6. 形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(building_mask, building_mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(building_mask, building_mask, cv::MORPH_OPEN, kernel);
    
    // 7. 计算法线图
    cv::Mat global_normal_map;
    cv::Mat smoothed_dsm = dsm_float.clone();
    
    if (use_dsine) {
        LOG_INFO << "Using DSINE for normal map...";
        global_normal_map = compute_normal_map_dsine_satellite_optimized(tdom_img, dsine_model_path);
    } else {
        LOG_INFO << "Using robust plane fitting for normal map...";
        global_normal_map = compute_normal_map_robust(dsm_float, 5);
    }
    
    // 8. 查找建筑轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(building_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    LOG_INFO << "Found " << contours.size() << " building contours";
    
    // 9. 处理每个建筑轮廓（使用增强的安全函数）
    int building_index = 0;
    std::map<int, std::pair<cv::Rect, double>> building_info;
    
    for (size_t i = 0; i < contours.size(); ++i) {
        if (building_index % 5 == 0) {
            log_memory_usage();
        }
        
        // 使用增强的函数处理每个建筑
        bool success = process_building_contour_enhanced(
            smoothed_dsm, tdom_img, global_normal_map, building_mask,
            dsm_gt, target_resolution, i, contours[i],
            building_index, buildings, building_info, output_dir
        );
        
        if (!success) {
            LOG_DEBUG << "Failed to process building contour " << i;
        }
    }
    
    LOG_INFO << "Successfully processed " << building_index << " buildings.";
    
    if (building_index == 0) {
        LOG_ERROR << "No valid buildings found in the region.";
        return;
    }
    
    // 10. 保存建筑索引信息
    std::ofstream info_file(output_dir + "/building_info.txt");
    if (info_file.is_open()) {
        info_file << "=== Building Information ===" << std::endl;
        info_file << "Processing Time: " << std::ctime(nullptr);
        info_file << "Total Buildings: " << building_index << std::endl;
        info_file << std::endl;
        
        for (const auto& [idx, info] : building_info) {
            const auto& bbox = info.first;
            double base_height = info.second;
            info_file << "Building " << idx << ":" << std::endl;
            info_file << "  Bounding Box: [" << bbox.x << ", " << bbox.y << ", " 
                     << bbox.width << ", " << bbox.height << "]" << std::endl;
            info_file << "  Base Height: " << base_height << " m" << std::endl;
            info_file << std::endl;
        }
        info_file.close();
        LOG_INFO << "Building info saved";
    }
}
/**
 * @brief 写入LOD模型
 */
void write_lod_models(int nb_buildings, const std::string& output_dir,
                     const std::vector<std::unique_ptr<UBlock_building>>& buildings) {
    if (nb_buildings == 0) {
        LOG_WARNING << "No buildings to write LOD models";
        return;
    }
    
    std::string prefix = fs::path(output_dir).stem().string();
    
    try {
        // 写入LOD2模型
        std::string lod2_file = output_dir + "/" + prefix + "_lod2.obj";
        std::ofstream lod2_stream(lod2_file);
        if (!lod2_stream.is_open()) {
            LOG_ERROR << "Failed to open LOD2 file: " << lod2_file;
        } else {
            // 这里需要根据UBlock_building的实际接口来写入模型
            // 假设每个建筑都有生成mesh的方法
            for (int i = 0; i < nb_buildings; ++i) {
                // buildings[i]->write_to_obj(lod2_stream); // 需要实现这个方法
            }
            LOG_INFO << "LOD2 model written: " << lod2_file;
            lod2_stream.close();
        }
        
        // 写入汇总信息
        std::string summary_file = output_dir + "/" + prefix + "_summary.txt";
        std::ofstream summary_stream(summary_file);
        if (summary_stream.is_open()) {
            summary_stream << "=== LOD Models Summary ===" << std::endl;
            summary_stream << "Generated at: " << std::ctime(nullptr);
            summary_stream << "Number of Buildings: " << nb_buildings << std::endl;
            summary_stream << "Output Directory: " << output_dir << std::endl;
            summary_stream << "LOD2 File: " << lod2_file << std::endl;
            summary_stream.close();
            LOG_INFO << "Summary saved to " << summary_file;
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR << "Error writing LOD models: " << e.what();
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        LOG_INFO << "Usage: " << argv[0] << " <config_directory>";
        LOG_INFO << "No configuration file.";
        return 1;
    }
    
    // 初始化GDAL
    GDALAllRegister();
    
    // 初始化Python解释器
    py::scoped_interpreter guard{};
    
    // 启动计时器
    boost::timer::auto_cpu_timer t("%w s\n");
    
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    std::cout << "GDAL version: " << GDALVersionInfo("RELEASE_NAME") << std::endl;

    const std::string wdir(argv[1]);

    // 初始化日志
    cm::initialize_logger(cm::severity_level::debug);
    
    // 加载配置
    cm::read_config(wdir + "/config_modeling.xml");
    LOG_INFO << "Configuration loaded from " << wdir << "/config_modeling.xml";

    try {
        auto& config = cm::get_config();
        const std::string dsm_file = config.get<std::string>("dsm_file");
        const std::string tdom_file = config.get<std::string>("tdom_file");
        const std::string segmentation_file = config.get<std::string>("segmentation_file");
        const std::string output_dir = config.get<std::string>("output_dir");
        const int rec_thread_num = std::stoi(config.get<std::string>("rec_thread_num"));
        const int cal_thread_num = std::stoi(config.get<std::string>("cal_thread_num"));
        
        // 可选参数
        double resolution = 0.5;
        double min_building_area = 100.0;
        bool use_dsine = false;
        std::string dsine_model_path = "";
        
        try {
            resolution = std::stod(config.get<std::string>("resolution", "0.5"));
            min_building_area = std::stod(config.get<std::string>("min_building_area", "100.0"));
            use_dsine = config.get<std::string>("use_dsine", "false") == "true";
            dsine_model_path = config.get<std::string>("dsine_model_path", "");
        } catch (...) {
            LOG_WARNING << "Using default parameters";
        }
        
        LOG_INFO << "Parameters:";
        LOG_INFO << "  Resolution: " << resolution << " m/px";
        LOG_INFO << "  Min Building Area: " << min_building_area << " m²";
        LOG_INFO << "  Use DSINE: " << (use_dsine ? "Yes" : "No");
        if (use_dsine) {
            LOG_INFO << "  DSINE Model: " << (dsine_model_path.empty() ? "Default" : dsine_model_path);
        }
        
        // 确保输出目录存在
        if (!fs::exists(output_dir)) {
            fs::create_directories(output_dir);
            LOG_INFO << "Created output directory: " << output_dir;
        }
        
        // 处理卫星数据（改进版）
        std::vector<std::unique_ptr<UBlock_building>> buildings;
        process_satellite_data_safe(
            dsm_file,
            tdom_file,
            segmentation_file,
            output_dir,
            buildings,
            use_dsine,
            dsine_model_path,
            resolution,
            min_building_area
        );
        
        const int nb_buildings = (int)buildings.size();
        if (nb_buildings == 0) {
            LOG_ERROR << "No buildings found in the region.";
            return 1;
        }
        
        LOG_INFO << "Starting building modeling for " << nb_buildings << " buildings...";
        clock_t start = clock();
        
        // 设置OpenMP线程数
        omp_set_num_threads(rec_thread_num);
        LOG_INFO << "Using " << rec_thread_num << " threads for building modeling";
        
        // 并行处理每个建筑
        #pragma omp parallel for
        for (int i = 0; i < nb_buildings; ++i) {
            LOG_INFO << "Processing building #" << i;
            try {
                buildings[i]->segment_arrangement_modeling();
                LOG_INFO << "Building #" << i << " modeling completed";
            } catch (const std::exception& e) {
                LOG_ERROR << "Error processing building " << i << ": " << e.what();
            }
        }
        
        clock_t end = clock();
        double BuildingTime = ((double)end - start) / CLOCKS_PER_SEC;
        LOG_INFO << "Building Modeling Time: " << BuildingTime << " s.";
        
        // 写入LOD模型
        write_lod_models(nb_buildings, output_dir, buildings);
        
        LOG_INFO << "=== Processing Completed ===";
        LOG_INFO << "Output directory: " << output_dir;
        LOG_INFO << "Total buildings processed: " << nb_buildings;
        LOG_INFO << "Total processing time: see above timer output";
        
    } catch (std::exception &e) {
        LOG_ERROR << "Error: " << e.what();
        return 1;
    }
    
    // 清理GDAL
    GDALDestroyDriverManager();
    
    return 0;
}
