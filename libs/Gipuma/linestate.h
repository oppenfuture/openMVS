#pragma once
#ifndef _GIPUMA_LINE_STATE_H_
#define _GIPUMA_LINE_STATE_H_
#include <string.h> // memset()
#include "algorithmparameters.h"
#include "cameraparameters.h"
#include "managed.h"
#include <vector_types.h> // float4

class __align__(128) LineState : public Managed {
public:
    float4 *norm4; // 3 values for normal and last for d
    float *c; // cost
    float *init_depth; // realsense and sparse pointcloud depth
    /*float *disp; // disparity*/
    int n;
    int s; // stride
    int l; // length
    void resize(int n)
    {
        cudaMallocManaged (&c,        sizeof(float) * n);
        cudaMallocManaged (&init_depth, sizeof(float) * n);
        /*cudaMallocManaged (&disp,     sizeof(float) * n);*/
        cudaMallocManaged (&norm4,    sizeof(float4) * n);
        memset            (c,      0, sizeof(float) * n);
        memset            (init_depth, 0, sizeof(float) * n);
        /*memset            (disp,   0, sizeof(float) * n);*/
        memset            (norm4,  0, sizeof(float4) * n);
    }
    ~LineState()
    {
        cudaFree (c);
        cudaFree (norm4);
        cudaFree (init_depth);
    }
};

#endif // _GIPUMA_LINE_STATE_H_