#pragma once

struct material
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

static __host__ material make_lambertian(const float3& albedo)
{
	return material{ albedo };
}

static __host__ material make_metal(const float3& albedo, float fuzz)
{
	return material{ albedo, fuzz };
}

static __host__ material make_dielectric(float ir)
{
	return material{ ir };
}

bool __forceinline__ __device__ near_zero(const float3& e) 
{
	// Return true if the vector is close to zero in all dimensions.
	const auto s = 1e-8;
	return (fabs(e.x) < s) && (fabs(e.y) < s) && (fabs(e.z) < s);
}

