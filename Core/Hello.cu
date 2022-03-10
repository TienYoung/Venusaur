#include <optix.h>

#include <cuda/helpers.h>

#include "Hello.h"

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
    params.image[launch_index.y * params.image_width + launch_index.x] =
        make_color(make_float3(rtData->r, rtData->g, rtData->b));
}