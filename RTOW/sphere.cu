#include <optix.h>

#include "sphere.h"

extern "C" {
__constant__ ParamsSphere params;
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
    auto ray_origin = params.camera_center;
    auto ray_direction = pixel_center - params.camera_center;

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            0,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2 );
    float3 result;
    result.x = __uint_as_float( p0 );
    result.y = __uint_as_float( p1 );
    result.z = __uint_as_float( p2 );

    write_color(params.image[j * image_width + i], result);
}

// extern "C"
// __global__ void __closesthit__()
// {
//     optixSetPayload_0( __float_as_uint( 1.0f ) );
//     optixSetPayload_1( __float_as_uint( 0.0f ) );
//     optixSetPayload_2( __float_as_uint( 0.0f ) );
// }

extern "C"
__global__ void __anyhit__()
{
    optixSetPayload_0( __float_as_uint( 1.0f ) );
    optixSetPayload_1( __float_as_uint( 0.0f ) );
    optixSetPayload_2( __float_as_uint( 0.0f ) );
}

extern "C"
__global__ void __miss__()
{
    auto ray_direction = optixGetWorldRayDirection();
    color pixel_color = ray_color(ray_direction);
    
    optixSetPayload_0( __float_as_uint( pixel_color.x ) );
    optixSetPayload_1( __float_as_uint( pixel_color.y ) );
    optixSetPayload_2( __float_as_uint( pixel_color.z ) );
}