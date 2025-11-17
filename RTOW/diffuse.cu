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
        .diffuse = {
            .x = __uint_as_float(optixGetPayload_2()),
            .y = __uint_as_float(optixGetPayload_3()),
            .z = __uint_as_float(optixGetPayload_4()),
        },
    };
}

static __forceinline__ __device__ void SetDiffusePayload(DiffusePayload payload)
{
    optixSetPayload_0(payload.seed);
    optixSetPayload_1(payload.depth);
    optixSetPayload_2(__float_as_uint(payload.diffuse.x));
    optixSetPayload_3(__float_as_uint(payload.diffuse.y));
    optixSetPayload_4(__float_as_uint(payload.diffuse.z));
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

    unsigned int samples_per_pixel = params.samples_per_pixel;
    
    unsigned int seed = tea<4>( j * image_width + i,  params.subframe_index);
    DiffusePayload payload = {
        .seed = seed,
        .depth = 31,
        .diffuse = make_float3(0.0f, 0.0f, 0.0f),
    };
    
    float3 result = make_float3(0.0f, 0.0f, 0.0f);
    for(unsigned int sample = 0; sample < samples_per_pixel; sample++)
    {
        const float2 offset = make_float2( rnd( seed ) - 0.5f, rnd( seed )- 0.5f );

        auto pixel_center = params.pixel00_loc + ((i + offset.x) * params.pixel_delta_u) + ((j + offset.y) * params.pixel_delta_v);
        auto ray_origin = params.camera_center;
        auto ray_direction = pixel_center - params.camera_center;

        unsigned int u0, u1, u2, u3, u4;
        u0 = payload.seed;
        u1 = payload.depth;
        u2 = __float_as_uint(payload.diffuse.x);
        u3 = __float_as_uint(payload.diffuse.y);
        u4 = __float_as_uint(payload.diffuse.z);
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
                u0, u1, u2, u3, u4);

        payload.seed = u0;
        payload.depth = u1;
        result.x += __uint_as_float( u2 );
        result.y += __uint_as_float( u3 );
        result.z += __uint_as_float( u4 );
    }

    write_color(params.image[j * image_width + i], result / samples_per_pixel);
}

extern "C"
__global__ void __closesthit__()
{
    DiffusePayload payload = GetDiffusePayload();
    --payload.depth;

    if (payload.depth == 0)
    {
        SetDiffusePayload(payload);
        return;
    }

    unsigned int seed = payload.seed;
    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    float3 w_in;
    cosine_sample_hemisphere(z1, z2, w_in);
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax()*ray_dir;
    
    payload.seed = seed;
    // payload.diffuse = w_in;

    // SetDiffusePayload(payload);

    unsigned int u0, u1, u2, u3, u4;
    u0 = payload.seed;
    u1 = payload.depth;
    u2 = __float_as_uint(payload.diffuse.x);
    u3 = __float_as_uint(payload.diffuse.y);
    u4 = __float_as_uint(payload.diffuse.z);
    optixTrace(
            params.handle,
            P,
            w_in,
            0.0f,                // Min intersection distance
            FLT_MAX,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask(255), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            0,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            u0, u1, u2, u3, u4);

    payload.seed = u0;
    payload.depth = u1;
    payload.diffuse.x += __uint_as_float( u2 );
    payload.diffuse.y += __uint_as_float( u3 );
    payload.diffuse.z += __uint_as_float( u4 );

    SetDiffusePayload(payload);
}

extern "C"
__global__ void __miss__()
{
    auto ray_direction = optixGetWorldRayDirection();
    float3 pixel_color = ray_color(ray_direction);
    
    SetDiffusePayload(DiffusePayload {
        .seed = optixGetPayload_0(),
        .depth = 0,
        .diffuse = pixel_color,
    });
}