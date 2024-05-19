#pragma once

struct Params
{
	uchar4* image;
	float4* accum;
	unsigned int width;
	unsigned int height;
	unsigned int samples_per_pixel;
	unsigned int subframe_index;

	float3 origin;
	float3 u, v, w;
	float lens_radius;

	OptixTraversableHandle handle;
};

struct SphereData
{
	float3 center;
	float radius;
};

struct MaterialData
{
	union
	{
		struct
		{
			float3 albedo;
			float fuzz;
		};
		float ir;
	};
};

struct RayGenData
{
};

struct MissData
{
};


struct SphereHitGroupData
{
};


