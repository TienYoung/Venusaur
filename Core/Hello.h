#pragma once

#include "vec_math.h"

#include "material.h"

struct Params
{
	float4* image;
	unsigned int image_width;
	unsigned int image_height;
	unsigned int samples_per_pixel;
	OptixTraversableHandle handle;
};

struct RayGenData
{
	float3 origin;
	float3 horizontal;
	float3 vertical;
	float3 lower_left_corner;
	float3 u, v, w;
	float lens_radius;
};

struct MissData
{
	float3 pixel_color;
};

struct SphereHitGroupData
{
	float3 center;
	float radius;
	material mat;
};