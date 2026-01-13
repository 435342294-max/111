cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(PROJ_NAME utils_apps)
project(${PROJ_NAME})

# pthread
find_package(Threads REQUIRED)

# OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
message(STATUS "Found OpenMP " ${OpenMP_VERSION})

# Find Boost
find_package(Boost REQUIRED COMPONENTS log program_options)

# Find CGAL libraries
find_package(CGAL REQUIRED COMPONENTS Core)
message(STATUS "Found CGAL " ${CGAL_VERSION})

# Find OpenSceneGraph
find_package(OpenSceneGraph REQUIRED COMPONENTS osgViewer osgText osgDB osgGA osgManipulator osgUtil)
include(UseOpenSceneGraph)

# Find OpenGL
find_package(OpenGL REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Assimp
find_package(assimp 5 REQUIRED CONFIG)

# Find MATLAB
find_package(Matlab REQUIRED COMPONENTS ENGINE_LIBRARY DATAARRAY_LIBRARY)
if (Matlab_FOUND AND NOT TARGET libMatlab)
  add_library(libMatlab INTERFACE)
  target_include_directories(libMatlab INTERFACE ${Matlab_INCLUDE_DIRS})
  target_link_libraries(libMatlab INTERFACE ${Matlab_ENGINE_LIBRARY} ${Matlab_DATAARRAY_LIBRARY})
  message(STATUS "Import libMatlab library.")
endif(Matlab_FOUND AND NOT TARGET libMatlab)

# Build and link RPLY
include(UseRPLY)

# Find glm
find_package(glm REQUIRED)

# Find GLFW
find_package(glfw3 3.2 REQUIRED)

# Build and link GLAD
include(UseGLAD)

# test_config executable
set(TARGET test_config)
add_executable(${TARGET} test_config.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Boost::boost
)

# test_flatmanipulator executable
set(TARGET test_flatmanipulator)
add_executable(${TARGET} test_flatmanipulator.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Boost::dynamic_linking
    Boost::log
    CGAL::CGAL
    libOSG
)

# test_logger executable
set(TARGET test_logger)
add_executable(${TARGET} test_logger.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Boost::dynamic_linking
    Boost::log
)
target_compile_definitions(${TARGET} PRIVATE BOOST_LOG_DYN_LINK)

# test_manhattan_detector executable
# TODO: why explicit pthread required
set(TARGET test_manhattan_detector)
add_executable(${TARGET} test_manhattan_detector.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Threads::Threads
    CGAL::CGAL
    libOSG
)

# test_matlab_ostream_helper
set(TARGET test_matlab_ostream_helper)
add_executable(${TARGET} test_matlab_ostream_helper.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    libMatlab
)

# test_model_loader
set(TARGET test_model_loader)
add_executable(${TARGET} test_model_loader.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Boost::dynamic_linking
    Boost::log
    CGAL::CGAL
    ${OpenCV_LIBS}
    assimp::assimp
    libRPLY
)

# test_gl_renderer executable
set(TARGET test_gl_renderer)
add_executable(${TARGET} test_gl_renderer.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Boost::dynamic_linking
    Boost::log
    Boost::filesystem
    Boost::program_options
    CGAL::CGAL
    ${OpenCV_LIBS}
    assimp::assimp
    libRPLY
    OpenGL::GL
    glm
    glfw
    libGLAD
)
target_compile_definitions(${TARGET} PRIVATE BOOST_LOG_DYN_LINK)

# test_index_property_map executable
set(TARGET test_index_property_map)
add_executable(${TARGET} test_index_property_map.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Boost::boost
)

# test_viewer executable
set(TARGET test_viewer)
add_executable(${TARGET} test_viewer.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Boost::dynamic_linking
    Boost::log
    CGAL::CGAL
    libOSG
)
target_compile_definitions(${TARGET} PRIVATE BOOST_LOG_DYN_LINK)

# offscreen rendering executable
set(TARGET offscreen_rendering)
add_executable(${TARGET} offscreen_rendering.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    Boost::dynamic_linking
    Boost::log
    CGAL::CGAL
    ${OpenCV_LIBS}
    assimp::assimp
    libRPLY
    OpenGL::GL
    glm
    glfw
    libGLAD
)
target_compile_definitions(${TARGET} PRIVATE BOOST_LOG_DYN_LINK)

# test_ransac executable
set(TARGET test_ransac)
add_executable(${TARGET} test_ransac.cpp)
target_link_libraries(${TARGET}
  PRIVATE
    utils
    OpenMP::OpenMP_CXX
    CGAL::CGAL
    ${OpenCV_LIBS}
)
