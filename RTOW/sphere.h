#pragma once

#include <params.h>

struct ParamsSphere : Params
{
    float3 camera_center;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;
    OptixTraversableHandle handle;
};

__forceinline__ __device__  color ray_color(const float3& ray_direction) {
    float3 unit_direction = unit_vector(ray_direction);
    auto a = 0.5f*(unit_direction.y + 1.0f);
    return (1.0f-a)*color(1.0f, 1.0f, 1.0f) + a*color(0.5f, 0.7f, 1.0f);
}