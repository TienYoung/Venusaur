#include <optix.h>

#include "Hello.h"

extern "C" 
{
    __constant__ Params params;
}

static __forceinline__ __device__ void setPayload(float3 p)
{
	optixSetPayload_0(__float_as_uint(p.x));
	optixSetPayload_1(__float_as_uint(p.y));
	optixSetPayload_2(__float_as_uint(p.z));
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

    auto u = double(launch_index.x) / (params.image_width - 1);
    auto v = double(launch_index.y) / (params.image_height - 1);

    float3 origin = rtData->origin;
    float3 direction = rtData->lower_left_corner + u * rtData->horizontal + v * rtData->vertical - rtData->origin;

	// Trace the ray against our scene hierarchy
	unsigned int p0, p1, p2;
	optixTrace(
		params.handle,
		origin,
		direction,
		0.0f,                // Min intersection distance
		1e16f,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_NONE,
		0,                   // SBT offset   -- See SBT discussion
		1,                   // SBT stride   -- See SBT discussion
		0,                   // missSBTIndex -- See SBT discussion
		p0, p1, p2);

	float3 pixel_color = make_float3(
		__uint_as_float(p0),
		__uint_as_float(p1),
		__uint_as_float(p2)
	);

    params.image[launch_index.y * params.image_width + launch_index.x] = make_float4(pixel_color, 1.0f);
}

extern "C" __global__ void __miss__ray_color()
{
    float3 unit_direction = normalize(optixGetWorldRayDirection());
    auto t = 0.5 * (unit_direction.y + 1.0);
    setPayload(lerp(make_float3(1.0), make_float3(0.5, 0.7, 1.0), t));
}