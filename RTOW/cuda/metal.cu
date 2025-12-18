#include <cuda/std/cstdint>
#include <optix.h>

#include "params.h"
#include "random.h"
#include "shading.h"

#include "metal.h"

static __forceinline__ __device__ void* unpackPointer(uint32_t i0, uint32_t i1)
{
    const uintptr_t uptr = static_cast<uintptr_t>(i0) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>(uptr); 
    return ptr;
}

static __forceinline__ __device__ void packPointer(void* ptr, uint32_t& i0, uint32_t& i1)
{
    const uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T* getPayload()
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
                1,                   // SBT stride   -- See SBT discussion
                0,                   // missSBTIndex -- See SBT discussion
                u0, u1);

            // optixReorder();

            optixInvoke(u0, u1);

            ray_origin = payload.origin;
            ray_direction = payload.direction;
        } 
        while(payload.depth != 0); 
        
        result += payload.diffuse;
    }

    write_color(params.image[j * image_width + i], result / samples_per_pixel);
}

union Sphere
{
    struct
    {
        float3 center;
        float radius;
    };
    float4 data;
};

extern "C" __global__ void __closesthit__lambertian()
{
    const float3 ray_origin    = optixGetWorldRayOrigin();
    const float3 ray_direction = optixGetWorldRayDirection();
    const float  ray_t         = optixGetRayTmax();
    const float3 hit_point     = ray_origin + ray_t * ray_direction;

    auto albedo = *reinterpret_cast<float3*>(optixGetSbtDataPointer());
    
    MetalPayload* payload = getPayload<MetalPayload>();

    if(optixGetPrimitiveType() != OPTIX_PRIMITIVE_TYPE_SPHERE)
    {
        payload->depth = 0;
        payload->diffuse *= 0.0f;
        return;
    }

    Sphere sphere;
    optixGetSphereData(&sphere.data);
    
    const float3 world_normal = (hit_point - sphere.center) / sphere.radius;

    const float z1 = rnd(payload->seed);
    const float z2 = rnd(payload->seed);
    float3 w_in;
    cosine_sample_hemisphere(z1, z2, w_in);
    Onb onb(world_normal);
    onb.inverse_transform(w_in);

    auto attenuation = --payload->depth > 0 ? albedo : float3{ .x = 0.0f, .y = 0.0f, .z = 0.0f };
    payload->origin = hit_point;
    payload->direction = w_in;
    payload->diffuse *= attenuation;
}


extern "C" __global__ void __miss__()
{
    MetalPayload* payload = getPayload<MetalPayload>();

    auto ray_direction = optixGetWorldRayDirection();
    float3 pixel_color = ray_color(ray_direction);
    
    payload->depth = 0;
    payload->diffuse *= pixel_color;
}