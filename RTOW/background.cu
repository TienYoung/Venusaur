#include <optix.h>

#include "background.h"

extern "C" {
__constant__ ParamsBackground params;
}

extern "C"
__global__ void __raygen__()
{
    uint3 launchIndex = optixGetLaunchIndex();
    uint3 launchDimensions = optixGetLaunchDimensions();
    int i = launchIndex.x;
    int j = launchIndex.y;
    int image_width = launchDimensions.x;
    // int image_height = launchDimensions.y;

    auto pixel_center = params.pixel00_loc + (i * params.pixel_delta_u) + (j * params.pixel_delta_v);
    auto ray_direction = pixel_center - params.camera_center;

    color pixel_color = ray_color(ray_direction);
    write_color(params.image[j * image_width + i], pixel_color);
}