#pragma once

#include <params.h>

struct DiffuseParams
{
    uchar4* image;
    float3 camera_center;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;
    uint32_t samples_per_pixel;
    uint32_t subframe_index;
    OptixTraversableHandle handle;
};

struct DiffusePayload
{
    uint32_t seed;
    uint32_t depth;
    float3 diffuse;
};
