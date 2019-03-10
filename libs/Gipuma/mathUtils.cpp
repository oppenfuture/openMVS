/*
 * some math helper functions
 */

#include "mathUtils.h"

//rounding of positive float values
#if defined(_WIN32)
float roundf(float val) {
	return floor(val + 0.5f);
}
#endif

/* get angle between two vectors in 3D
 * Input: v1,v2 - vectors
 * Output: angle in radian
 */
float getAngle(cv::Vec3f v1, cv::Vec3f v2) {
    float angle = acosf(v1.dot(v2));
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;

    return angle;
}