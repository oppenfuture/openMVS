/*
 *  cameraGeometryUtils.h
 *
 * utility functions for camera geometry related stuff
 * most of them from: "Multiple View Geometry in computer vision" by Hartley and Zisserman
 */

#pragma once
#ifndef _GIPUMA_CAMERA_GEOMETRY_UTILS_H_
#define _GIPUMA_CAMERA_GEOMETRY_UTILS_H_
#include "mathUtils.h"
#include "main.h"
#include "camera.h"
#include "cameraparameters.h"
#include <vector_types.h>
#include <limits>
#include <signal.h>

cv::Mat_<float> getColSubMat(cv::Mat_<float> M, int* indices, int numCols);

// Multi View Geometry, page 163
cv::Mat_<float> getCameraCenter(cv::Mat_<float> &P);

inline cv::Vec3f get3Dpoint(Camera &cam, float x, float y, float depth);

inline cv::Vec3f get3Dpoint(Camera &cam, int x, int y, float depth);

// get the viewing ray for a pixel position of the camera
cv::Vec3f getViewVector(Camera &cam, int x, int y);

float getDepth(cv::Vec3f &X, cv::Mat_<float> &P);

cv::Mat_<float> getTransformationMatrix(cv::Mat_<float> R, cv::Mat_<float> t);

/* compute depth value from disparity or disparity value from depth
 * Input:  f         - focal length in pixel
 *         baseline  - baseline between cameras (in meters)
 *         d - either disparity or depth value
 * Output: either depth or disparity value
 */
float disparityDepthConversion(float f, float baseline, float d);

cv::Mat_<float> getTransformationReferenceToOrigin(cv::Mat_<float> R, cv::Mat_<float> t);

void transformCamera(cv::Mat_<float> R, cv::Mat_<float> t, cv::Mat_<float> transform, Camera &cam, cv::Mat_<float> K);

cv::Mat_<float> scaleK(cv::Mat_<float> K, float scaleFactor);
void copyOpencvVecToFloat4(cv::Vec3f &v, float4 *a);
void copyOpencvVecToFloatArray(cv::Vec3f &v, float *a);
void copyOpencvMatToFloatArray(cv::Mat_<float> &m, float **a);

/* get camera parameters (e.g. projection matrices) from file
 * Input:  inputFiles  - paths to calibration files
 *         scaleFactor - if image was rescaled we need to adapt calibration matrix K accordingly
 * Output: camera parameters
 */
CameraParameters getCameraParameters(CameraParameters_cu& cpc,
                                     const std::vector<cv::Mat_<float>> &projection_matrices,
                                     float scaleFactor = 1.0f,
                                     bool transformP = true);

#endif // _GIPUMA_CAMERA_GEOMETRY_UTILS_H_