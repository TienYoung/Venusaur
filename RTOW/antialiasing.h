#pragma once

#include <params.h>

struct AntialiasingParams
{
    uchar4* image;
    float3 camera_center;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;
    unsigned int samples_per_pixel;
    OptixTraversableHandle handle;
};