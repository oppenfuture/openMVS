//
// utility functions for reading and writing files
//
#include <iostream>
#include <fstream>

#include "fileIoUtils.h"

bool savePfm(const cv::Mat_<float> &image, const std::string filePath, float scale_factor) {
    // Open the file as binary
    std::ofstream imageFile(filePath.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

    if (imageFile) {
        int width = image.cols;
        int height = image.rows;
        int numberOfComponents = image.channels();

        // write the type of PFM file
        char type[3];
        type[0] = 'P';
        type[2] = 0x0a;

        if (numberOfComponents == 1) {
            type[1] = 'f';
        } else if (numberOfComponents == 3) {
            type[1] = 'F';
        }

        imageFile << type[0] << type[1] << type[2];

        // write the width and height
        imageFile << width << ' ' << height << type[2];

        // write the scale factor and assume little endian storage
        imageFile << -scale_factor << type[2];

        // write the floating points grayscale or rgb color from bottom to top, left to right
        float* buffer = new float[numberOfComponents];

        for (int i=0; i<height; ++i) {
            for (int j=0; j<width; ++j) {
                if (numberOfComponents == 1) {
                    buffer[0] = image(height-1-i, j);
                } else {
                    cv::Vec3f color = image(height-1-i, j);

                    // OpenCV store as BGR
                    buffer[0] = color.val[2];
                    buffer[1] = color.val[1];
                    buffer[2] = color.val[0];
                }

                imageFile.write((char *)buffer, numberOfComponents*sizeof(float));
            }
        }

        delete [] buffer;

        std::cout << "Save " << filePath << std::endl;

        imageFile.close();
    } else {
        std::cerr << "Could not open the file : " << filePath << std::endl;
        return false;
    }

    return true;
}
