//
// interface of gipuma
//

#pragma once
#ifndef _GIPUMA_GIPUMA_MAIN_H_
#define _GIPUMA_GIPUMA_MAIN_H_

#include <vector>
#include <opencv2/opencv.hpp>

namespace gipuma {
void selectCudaDevice();

bool GipumaMain(
  const std::string &prefix,
  const std::vector<cv::Mat_<float>> &images,
        const std::vector<cv::Mat_<float>> &projection_matrices,
        cv::Mat_<float> &depth_map,
        cv::Mat_<cv::Point3_<float>> &normal_map,
        float &dMin, float &dMax,
        const char* config_filename,
        bool importReferenceDepth);
}
#endif // _GIPUMA_GIPUMA_MAIN_H_
