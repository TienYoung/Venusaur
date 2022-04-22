#pragma once

struct SphereHitGroupData
{
	float3 center;
	float radius;
	material mat;
};

static __host__ OptixAabb genSphere(SphereHitGroupData& sphere, const float3& center, float radius, const material& mat)
{
	sphere.center = center;
	sphere.radius = radius;
	sphere.mat = mat;
	return { center.x - radius, center.y - radius, center.z - radius, center.x + radius, center.y + radius, center.z + radius };
}
