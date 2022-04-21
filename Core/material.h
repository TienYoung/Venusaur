#pragma once

struct material
{
	float3 albedo;
	float fuzz;
};

bool __forceinline__ __device__ near_zero(const float3& e) 
{
	// Return true if the vector is close to zero in all dimensions.
	const auto s = 1e-8;
	return (fabs(e.x) < s) && (fabs(e.y) < s) && (fabs(e.z) < s);
}

