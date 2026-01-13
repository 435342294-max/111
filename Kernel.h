#ifndef KERNEL_H
#define KERNEL_H

#include <CGAL/Simple_cartesian.h>

typedef CGAL::Simple_cartesian<double> Kernel;

// common typedefs
typedef Kernel::Point_3 Point_3;
typedef Kernel::Vector_3 Vector_3;
typedef Kernel::Plane_3 Plane_3;

#endif // KERNEL_H
