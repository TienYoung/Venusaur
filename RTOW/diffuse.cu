#include <cfloat>

#include <optix.h>

#include <random.h>
#include <shading.h>

#include "diffuse.h"

static __forceinline__ __device__ DiffusePayload GetDiffusePayload()
{
    return DiffusePayload {
        .seed = optixGetPayload_0(),
        .depth = optixGetPayload_1(),
        .origin = {
            .x = __uint_as_float(optixGetPayload_2()),
            .y = __uint_as_float(optixGetPayload_3()),
            .z = __uint_as_float(optixGetPayload_4()),
        },
        .direction = {
            .x = __uint_as_float(optixGetPayload_5()),
            .y = __uint_as_float(optixGetPayload_6()),
            .z = __uint_as_float(optixGetPayload_7()),
        },
        .diffuse = {
            .x = __uint_as_float(optixGetPayload_8()),
            .y = __uint_as_float(optixGetPayload_9()),
            .z = __uint_as_float(optixGetPayload_10()),
        },
    };
}

static __forceinline__ __device__ void SetDiffusePayload(DiffusePayload payload)
{
    optixSetPayload_0(payload.seed);
    optixSetPayload_1(payload.depth);
    optixSetPayload_2(__float_as_uint(payload.origin.x));
    optixSetPayload_3(__float_as_uint(payload.origin.y));
    optixSetPayload_4(__float_as_uint(payload.origin.z));
    optixSetPayload_5(__float_as_uint(payload.direction.x));
    optixSetPayload_6(__float_as_uint(payload.direction.y));
    optixSetPayload_7(__float_as_uint(payload.direction.z));
    optixSetPayload_8(__float_as_uint(payload.diffuse.x));
    optixSetPayload_9(__float_as_uint(payload.diffuse.y));
    optixSetPayload_10(__float_as_uint(payload.diffuse.z));
}

extern "C" {
__constant__ DiffuseParams params;
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
    
    float3 result = float3{.x = 0.0f, .y = 0.0f, .z = 0.0f };
    uint32_t seed = tea<4>( j * image_width + i,  params.subframe_index);
    uint32_t samples_per_pixel = params.samples_per_pixel;
    for(uint32_t sample = 0; sample < samples_per_pixel; sample++)
    {
        const float2 offset = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        auto pixel_center = params.pixel00_loc + ((i + offset.x) * params.pixel_delta_u) + ((j + offset.y) * params.pixel_delta_v);
        auto ray_origin = params.camera_center;
        auto ray_direction = pixel_center - params.camera_center;

        DiffusePayload payload = {
            .seed = seed,
            .depth = 50,
            .origin = ray_origin,
            .direction = ray_direction,
            .diffuse = make_float3(1.0f, 1.0f, 1.0f),
        };

        do
        {
            uint32_t u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10;
            u0 = payload.seed;
            u1 = payload.depth;
            u2 = __float_as_uint(payload.origin.x);
            u3 = __float_as_uint(payload.origin.y);
            u4 = __float_as_uint(payload.origin.z);
            u5 = __float_as_uint(payload.direction.x);
            u6 = __float_as_uint(payload.direction.y);
            u7 = __float_as_uint(payload.direction.z);
            u8 = __float_as_uint(payload.diffuse.x);
            u9 = __float_as_uint(payload.diffuse.y);
            u10 = __float_as_uint(payload.diffuse.z);

            optixTraverse(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    FLT_MIN,             // Min intersection distance
                    FLT_MAX,             // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask(255), // Specify always visible
                    OPTIX_RAY_FLAG_NONE,
                    0,                   // SBT offset   -- See SBT discussion
                    0,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10);

            optixInvoke(u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10);

            payload.seed = u0;
            payload.depth = u1;
            payload.origin = {
                .x = __uint_as_float(u2),
                .y = __uint_as_float(u3),
                .z = __uint_as_float(u4),
            };
            payload.direction = {
                .x = __uint_as_float(u5),
                .y = __uint_as_float(u6),
                .z = __uint_as_float(u7),
            };
            payload.diffuse = {
                .x = __uint_as_float(u8),
                .y = __uint_as_float(u9),
                .z = __uint_as_float(u10),
            };

            ray_origin = payload.origin;
            ray_direction = payload.direction;
        } 
        while(payload.depth != 0); 
        
        result += payload.diffuse;
    }

    write_color(params.image[j * image_width + i], result / samples_per_pixel);
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

    DiffusePayload payload = GetDiffusePayload();

    const float z1 = rnd(payload.seed);
    const float z2 = rnd(payload.seed);

    float3 w_in;
    cosine_sample_hemisphere(z1, z2, w_in);
    Onb onb( world_normal );
    onb.inverse_transform( w_in );
    // const float3 ray_dir = optixGetWorldRayDirection();
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    payload.depth--;
    payload.origin = P;
    payload.direction = w_in;
    payload.diffuse *= payload.depth > 0 ? 0.5f : 0.0f;

    SetDiffusePayload(payload);
}

extern "C"
__global__ void __miss__()
{
    DiffusePayload payload = GetDiffusePayload();

    auto ray_direction = optixGetWorldRayDirection();
    float3 pixel_color = ray_color(ray_direction);
    
    payload.depth = 0;
    payload.diffuse *= pixel_color;

    SetDiffusePayload(payload);
}