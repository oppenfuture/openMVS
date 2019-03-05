#pragma once
#ifndef _GIPUMA_IMAGE_INFO_H_
#define _GIPUMA_IMAGE_INFO_H_
#include "managed.h"

class __align__(128) ImageInfo : public Managed{
public:
    // Image size
    int cols;
    int rows;

    // Total number of pixels
    int np;

    // Total number of bytes (may be different when padded)
    int nb;
};

#endif // _GIPUMA_IMAGE_INFO_H_