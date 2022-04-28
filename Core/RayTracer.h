#pragma once

struct Params
{
	uchar4* image;
	unsigned int width;
	unsigned int height;
	unsigned int samples_per_pixel;
	unsigned int subframe_index;

	float3 origin;
	float3 u, v, w;
	float lens_radius;

	OptixTraversableHandle handle;
};

struct RayGenData
{
};

struct MissData
{
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

struct SphereHitGroupData
{
	float3 center;
	float radius;
	MaterialData mat;
};

bool __forceinline__ __device__ near_zero(const float3& e)
{
	// Return true if the vector is close to zero in all dimensions.
	const auto s = 1e-8;
	return (fabs(e.x) < s) && (fabs(e.y) < s) && (fabs(e.z) < s);
}

