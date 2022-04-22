#pragma once

#include "rtweekend.h"

class camera
{
public:
	camera(float3 lookfrom, float3 lookat, float3 vup, float vfov, float aspect_ratio)
	{
		auto theta = degrees_to_radians(vfov);
		auto h = tanf(theta / 2);
		auto viewport_height = 2.0f * h;
		auto viewport_width = aspect_ratio * viewport_height;
		
		auto w = normalize(lookfrom - lookat);
		auto u = normalize(cross(vup, w));
		auto v = cross(w, u);

		origin = lookfrom;
		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
	}

	void set_sbt(RayGenSbtRecord& rg_sbt)
	{
		rg_sbt.data.origin = origin;
		rg_sbt.data.horizontal = horizontal;
		rg_sbt.data.vertical = vertical;
		rg_sbt.data.lower_left_corner = lower_left_corner;
	}

private:
	float3 origin;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
};
