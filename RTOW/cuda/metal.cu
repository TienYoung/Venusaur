#include <cuda/std/cstdint>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <vector_functions.h>

#include "params.h"
#include "random.h"
#include "shading.h"

#include "metal.h"

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, unsigned int i1)
{
    const uintptr_t uptr = static_cast<uintptr_t>(i0) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>(uptr); 
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T* GetPayload()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

extern "C" __device__ float3 __direct_callable__lambertian__(float3 ray_direction, float3 normal)
{
    return normal;
}

extern "C" __device__ float3 __direct_callable__metal__(float3 ray_direction, float3 normal)
{
    return normal;
}

extern "C" 
{
    __constant__ MetalParams params;
}

extern "C" __global__ void __raygen__()
{
    uint3 launchIndex = optixGetLaunchIndex();
    uint3 launchDimensions = optixGetLaunchDimensions();
    int i = launchIndex.x;
    int j = launchIndex.y;
    int image_width = launchDimensions.x;
    // int image_height = launchDimensions.y;
    
    float3 result = float3{.x = 0.0f, .y = 0.0f, .z = 0.0f };
    unsigned int seed = tea<4>( j * image_width + i,  params.subframe_index);
    unsigned int samples_per_pixel = params.samples_per_pixel;
    for(unsigned int sample = 0; sample < samples_per_pixel; sample++)
    {
        const float2 offset = make_float2(rnd(seed) - 0.5f, rnd(seed) - 0.5f);

        auto pixel_center = params.pixel00_loc + ((i + offset.x) * params.pixel_delta_u) + ((j + offset.y) * params.pixel_delta_v);
        auto ray_origin = params.camera_center;
        auto ray_direction = pixel_center - params.camera_center;

        MetalPayload payload = {
            .seed = seed,
            .depth = 50,
            .origin = ray_origin,
            .direction = ray_direction,
            .diffuse = make_float3(1.0f, 1.0f, 1.0f),
        };

        do
        {
            unsigned int u0, u1;
            packPointer(&payload, u0, u1);

            optixTraverse(
                    params.handle,
                    ray_origin,
                    ray_direction,
                    0.001f,             // Min intersection distance
                    10000000.0f,             // Max intersection distance
                    0.0f,                // rayTime -- used for motion blur
                    OptixVisibilityMask(255), // Specify always visible
                    OPTIX_RAY_FLAG_NONE,
                    0,                   // SBT offset   -- See SBT discussion
                    0,                   // SBT stride   -- See SBT discussion
                    0,                   // missSBTIndex -- See SBT discussion
                    u0, u1);

            optixInvoke(u0, u1);

            ray_origin = payload.origin;
            ray_direction = payload.direction;
        } 
        while(payload.depth != 0); 
        
        result += payload.diffuse;
    }

    write_color(params.image[j * image_width + i], result / samples_per_pixel);
}

extern "C" __global__ void __closesthit__()
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

    MetalPayload* payload = GetPayload<MetalPayload>();

    const float z1 = rnd(payload->seed);
    const float z2 = rnd(payload->seed);

    float3 w_in;
    cosine_sample_hemisphere(z1, z2, w_in);
    Onb onb( world_normal );
    onb.inverse_transform( w_in );
    // const float3 ray_dir = optixGetWorldRayDirection();
    const float3 P = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;

    payload->depth--;
    payload->origin = P;
    payload->direction = w_in;
    payload->diffuse *= payload->depth > 0 ? 0.5f : 0.0f;
}


extern "C" __global__ void __miss__()
{
    MetalPayload* payload = GetPayload<MetalPayload>();

    auto ray_direction = optixGetWorldRayDirection();
    float3 pixel_color = ray_color(ray_direction);
    
    payload->depth = 0;
    payload->diffuse *= pixel_color;
}