//
// utility functions for reading and writing files
//
#pragma once
#ifndef _GIPUMA_FILE_IO_UTILS_H_
#define _GIPUMA_FILE_IO_UTILS_H_
#include <string>
#include <opencv2/opencv.hpp>

bool savePfm(const cv::Mat_<float> &image, const std::string filePath, float scale_factor=1);

#endif // _GIPUMA_FILE_IO_UTILS_H_