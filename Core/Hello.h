#pragma once

#include "vec_math.h"

struct Params
{
	float4* image;
	unsigned int image_width;
	unsigned int image_height;
	OptixTraversableHandle handle;
};

struct RayGenData
{
	float3 origin;
	float3 horizontal;
	float3 vertical;
	float3 lower_left_corner;
};

struct MissData
{
	float3 pixel_color;
};