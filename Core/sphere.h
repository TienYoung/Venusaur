#pragma once

struct SphereHitGroupData
{
	float3 center;
	float radius;
	material mat;
};

static __host__ SphereHitGroupData make_sphere( const float3& center, float radius, const material& mat)
{
	return { center, radius, mat };
}

static __host__ OptixAabb gen_aabb(const SphereHitGroupData& sphere)
{
	float radius = fabsf(sphere.radius);
	return { 
		sphere.center.x - radius, 
		sphere.center.y - radius,
		sphere.center.z - radius, 
		sphere.center.x + radius, 
		sphere.center.y + radius, 
		sphere.center.z + radius 
	};
}
