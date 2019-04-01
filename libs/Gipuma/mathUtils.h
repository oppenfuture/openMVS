/*
 * some math helper functions
 */

#pragma once
#ifndef _GIPUMA_MATH_UTILS_H_
#define _GIPUMA_MATH_UTILS_H_
#include <opencv2/opencv.hpp>

#ifndef M_PI
#define M_PI    3.14159265358979323846f
#endif
#define M_PI_float    3.14159265358979323846f

/* get angle between two vectors in 3D
 * Input: v1,v2 - vectors
 * Output: angle in radian
 */
float getAngle(cv::Vec3f v1, cv::Vec3f v2);

#endif // _GIPUMA_MATH_UTILS_H_