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

static __host__ void genLambertianMat(material& mat, const float3& albedo)
{
	mat.albedo = albedo;
}

static __host__ void genMetalMat(material& mat, const float3& albedo, float fuzz)
{
	mat.albedo = albedo;
	mat.fuzz;
}

static __host__ void genDieletricMat(material& mat, float ir)
{
	mat.ir = ir;
}

bool __forceinline__ __device__ near_zero(const float3& e) 
{
	// Return true if the vector is close to zero in all dimensions.
	const auto s = 1e-8;
	return (fabs(e.x) < s) && (fabs(e.y) < s) && (fabs(e.z) < s);
}

