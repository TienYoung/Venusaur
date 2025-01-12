#include <optix.h>

#include "rt.h"
#include <cuda/helpers.h>

extern "C" {
	__constant__ Params params;
}

extern "C" __global__ void __raygen__rg()
{
	uint3 launch_index = optixGetLaunchIndex();
	RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

	params.image[launch_index.y * params.image_width + launch_index.x] =
		make_color(make_float3(rtData->r, rtData->g, rtData->b));
}

extern "C" __global__ void __miss__ms()
{
	uint3 launch_index = optixGetLaunchIndex();
	RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

	params.image[launch_index.y * params.image_width + launch_index.x] =
		make_color(make_float3(rtData->r, rtData->g, rtData->b));
}