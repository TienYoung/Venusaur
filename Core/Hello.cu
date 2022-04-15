#include <optix.h>

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
    params.image[launch_index.y * params.image_width + launch_index.x] = make_float4(static_cast<float>(launch_index.x) / params.image_width, (float)launch_index.y / params.image_height, 0.25f, 1.0f);
}