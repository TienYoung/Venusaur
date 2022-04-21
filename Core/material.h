#pragma once

struct material
{
	enum type
	{
		lambertian,
		metal
	};

	type ty;
	float3 albedo;
};

bool __forceinline__ __device__ near_zero(const float3& e) 
{
	// Return true if the vector is close to zero in all dimensions.
	const auto s = 1e-8;
	return (fabs(e.x) < s) && (fabs(e.y) < s) && (fabs(e.z) < s);
}

