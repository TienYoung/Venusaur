﻿#include <optix.h>

#include <OptiXToolkit/ShaderUtil/color.h>

#include "rt.h"

extern "C" {
	__constant__ Params params;
}

extern "C" __global__ void __raygen__rg()
{
	const uint3 launch_index = optixGetLaunchIndex();
	int i = launch_index.x;
	int j = launch_index.y;

	int image_width = params.width;
	int image_height = params.height;

	auto r = double(i) / (image_width - 1);
	auto g = double(j) / (image_height - 1);
	auto b = 0.0;

	params.image[j * image_width + i] = make_color(make_float3(r, g, b));
}

extern "C" __global__ void __miss__ms()
{
	//uint3 launch_index = optixGetLaunchIndex();
	//RayGenData* rtData = (RayGenData*)optixGetSbtDataPointer();

	//auto a = 0.5 * launch_index.y / params.image_width + 1.0;
	//params.image[launch_index.y * params.image_width + launch_index.x] = make_color((1.0 - a) * make_float3(1.0, 1.0, 1.0) + a * make_float3(0.5, 0.7, 1.0));
}