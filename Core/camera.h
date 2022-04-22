#pragma once

#include "rtweekend.h"

class camera
{
public:
	camera(float3 lookfrom, float3 lookat, float3 vup, float vfov, float aspect_ratio, float aperture, float focus_dist)
	{
		auto theta = degrees_to_radians(vfov);
		auto h = tanf(theta / 2);
		auto viewport_height = 2.0f * h;
		auto viewport_width = aspect_ratio * viewport_height;
		
		w = normalize(lookfrom - lookat);
		u = normalize(cross(vup, w));
		v = cross(w, u);

		origin = lookfrom;
		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - focus_dist * w;
		lens_radius = aperture / 2;
	}

	void set_sbt(RayGenSbtRecord& rg_sbt)
	{
		rg_sbt.data.origin = origin;
		rg_sbt.data.horizontal = horizontal;
		rg_sbt.data.vertical = vertical;
		rg_sbt.data.lower_left_corner = lower_left_corner;
		rg_sbt.data.u = u;
		rg_sbt.data.v = v;
		rg_sbt.data.w = w;
		rg_sbt.data.lens_radius = lens_radius;
	}

private:
	float3 origin;
	float3 lower_left_corner;
	float3 horizontal;
	float3 vertical;
	float3 u, v, w;
	float lens_radius;
};
