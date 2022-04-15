#include <optix.h>

struct Params
{
    float4* image;
    unsigned int image_width;
};

struct RayGenData
{
    float r, g, b;
};

extern "C" {
    __constant__ Params params;
}

extern "C"
__global__ void __raygen__draw_solid_color()
{
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();
    params.image[launch_index.y * params.image_width + launch_index.x] = make_float4(rtData->r, rtData->g, rtData->b, 1.0f);
}