#include <cfloat>

#include <optix.h>

#include "sphere.h"

extern "C" {
__constant__ SphereParams params;
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

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    
    auto pixel_center = params.pixel00_loc + (i * params.pixel_delta_u) + (j * params.pixel_delta_v);
    auto ray_origin = params.camera_center;
    auto ray_direction = pixel_center - params.camera_center;
    
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,                // Min intersection distance
        FLT_MAX,               // Max intersection distance
        0.0f,                // rayTime -- used for motion blur
        OptixVisibilityMask(255), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset   -- See SBT discussion
        0,                   // SBT stride   -- See SBT discussion
        0,                   // missSBTIndex -- See SBT discussion
        p0, p1, p2 );
        
    float3 result = {
        .x = __uint_as_float(p0),
        .y = __uint_as_float(p1),
        .z = __uint_as_float(p2),
    };

    write_color(params.image[j * image_width + i], result);
}

extern "C"
__global__ void __closesthit__()
{
    float t_hit = optixGetRayTmax();
    const float3 ray_orig = optixGetWorldRayOrigin();
    const float3 ray_dir  = optixGetWorldRayDirection();
    
    const unsigned int           prim_idx    = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
    const unsigned int           sbtGASIndex = optixGetSbtGASIndex();
    
    float4 q;
    // sphere center (q.x, q.y, q.z), sphere radius q.w
    optixGetSphereData( gas, prim_idx, sbtGASIndex, 0.f, &q );
    
    float3 world_raypos = ray_orig + t_hit * ray_dir;
    float3 obj_raypos   = optixTransformPointFromWorldToObjectSpace( world_raypos );
    float3 obj_normal   = ( obj_raypos - make_float3( q.x, q.y, q.z ) ) / q.w;
    float3 world_normal = unit_vector( optixTransformNormalFromObjectToWorldSpace( obj_normal ) );

    optixSetPayload_0( __float_as_uint(0.5f * (world_normal.x + 1.0f)));
    optixSetPayload_1( __float_as_uint(0.5f * (world_normal.y + 1.0f)));
    optixSetPayload_2( __float_as_uint(0.5f * (world_normal.z + 1.0f)));
}

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
    float3 pixel_color = ray_color(ray_direction);
    
    optixSetPayload_0( __float_as_uint( pixel_color.x ) );
    optixSetPayload_1( __float_as_uint( pixel_color.y ) );
    optixSetPayload_2( __float_as_uint( pixel_color.z ) );
}