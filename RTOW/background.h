#pragma once

#include <params.h>

struct ParamsBackground : Params
{
    float3 camera_center;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;
};

__forceinline__ __device__  color ray_color(const float3& ray_direction) {
    float3 unit_direction = unit_vector(ray_direction);
    auto a = 0.5*(unit_direction.y + 1.0);
    return (1.0-a)*color(1.0, 1.0, 1.0) + a*color(0.5, 0.7, 1.0);
}