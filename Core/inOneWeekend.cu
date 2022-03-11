#include <optix.h>

#include <vector_functions.h>
#include <vector_types.h>

#include "inOneWeekend.h"

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
    uint3 launch_dimensions = optixGetLaunchDimensions();
    RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

    auto i = launch_index.x;
    auto j = launch_index.y;

    const int image_width = launch_dimensions.x;
    const int image_height = launch_dimensions.y;


    auto r = double(i) / (image_width - 1);
    auto g = double(j) / (image_height - 1);
    auto b = 0.25;

    int ir = static_cast<int>(255.999 * r);
    int ig = static_cast<int>(255.999 * g);
    int ib = static_cast<int>(255.999 * b);

    params.image[launch_index.y * image_width + launch_index.x] = make_uchar3(ir, ig, ib);
}