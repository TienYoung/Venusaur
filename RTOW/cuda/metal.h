#pragma once

struct MetalParams {
    uchar4* image;
    float3 camera_center;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;
    unsigned int samples_per_pixel;
    unsigned int subframe_index;
    OptixTraversableHandle handle;
};

struct MetalPayload {
    unsigned int seed;
    unsigned int depth;
    float3 origin;
    float3 direction;
    float3 diffuse;
};

union Material {
    struct Lambertian {
        float3 albedo;
    } lambertian;

    struct Metal {
        float3 albedo;
        float fuzz;
    } metal;
};