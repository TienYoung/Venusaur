#include <optix.h>

#include "image.h"

extern "C" {
__constant__ ImageParams params;
}

extern "C"
__global__ void __raygen__()
{
    uint3 launchIndex = optixGetLaunchIndex();
    uint3 launchDimensions = optixGetLaunchDimensions();
    int i = launchIndex.x;
    int j = launchIndex.y;
    int image_width = launchDimensions.x;
    int image_height = launchDimensions.y;

    auto r = double(i) / (image_width-1);
    auto g = double(j) / (image_height-1);
    auto b = 0.0;

    int ir = int(255.999 * r);
    int ig = int(255.999 * g);
    int ib = int(255.999 * b);

    params.image[j * image_width + i] = {
        static_cast<unsigned char>(ir),
        static_cast<unsigned char>(ig), 
        static_cast<unsigned char>(ib), 
        255,
    };
}