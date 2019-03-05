#pragma once
#ifndef _GIPUMA_MAIN_H_
#define _GIPUMA_MAIN_H_
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#if CV_MAJOR_VERSION == 3
#include "opencv2/core/utility.hpp"
#endif

#include <omp.h>
#include <stdint.h>

typedef cv::Vec<uint16_t, 2> Vec2us;

struct Camera {
    Camera () : P (cv::Mat::eye(3,4,CV_32F)),  R(cv::Mat::eye ( 3,3,CV_32F ) ),baseline (0.54f), reference ( false ), depthMin ( 2.0f ), depthMax ( 20.0f ) {}
    cv::Mat_<float> P;
    cv::Mat_<float> P_inv;
    cv::Mat_<float> M_inv;
    //Mat_<float> K;
    cv::Mat_<float> R;
    cv::Mat_<float> R_orig_inv;
    cv::Mat_<float> t;
    cv::Vec3f C;
    float baseline;
    bool reference;
    float depthMin; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    float depthMax; //this could be figured out from the bounding volume (not done right now, but that's why this parameter is here as well and not only in AlgorithmParameters)
    //int id; //corresponds to the image name id (eg. 0-10), independent of order in argument list, just dependent on name
    std::string id;
    cv::Mat_<float> K;
    cv::Mat_<float> K_inv;
    //float f;
};

//parameters for camera geometry setup (assuming that K1 = K2 = K, P1 = K [I | 0] and P2 = K [R | t])
struct CameraParameters {
    CameraParameters () : rectified ( false ), idRef ( 0 ) {}
    cv::Mat_<float> K; //if K varies from camera to camera: K and f need to be stored within Camera
    cv::Mat_<float> K_inv; //if K varies from camera to camera: K and f need to be stored within Camera
    float f;
    bool rectified;
    std::vector<Camera> cameras;
    int idRef;
    std::vector<int> viewSelectionSubset;
};

struct Plane {
    cv::Mat_<cv::Vec3f> normal;
    cv::Mat_<float> d;
    void release () {
        normal.release ();
        d.release ();
    }
};

#endif // _GIPUMA_MAIN_H_
