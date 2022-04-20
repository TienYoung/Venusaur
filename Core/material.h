#pragma once

struct ray
{
	float3 origin;
	float3 direction;
};

struct material
{
	enum type
	{
		lambertian,
		metal
	};

	float3 albedo;
	type ty;
};

bool near_zero(const float3& e) const 
{
	// Return true if the vector is close to zero in all dimensions.
	const auto s = 1e-8;
	return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}

static __forceinline__ __device__ bool scatter(const material* mat, const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
{
	switch (mat->ty)
	{
	case material::lambertian:
		auto scatter_direction = rec.normal + random_unit_vector();
		// Catch degenerate scatter direction
		if (near_zero(scatter_direction))
			scatter_direction = rec.normal;
		scattered = ray{ rec.p, scatter_direction };
		attenuation = albedo;
		return true;

	case material::metal:
		float3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
		scattered = ray{ rec.p, reflected };
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	default:
		return false;
	}
}