#pragma once

class camera
{
public:
	camera()
	{
		auto aspect_ratio = 16.0f / 9.0f;
		auto viewport_height = 2.0f;
		auto viewport_width = aspect_ratio * viewport_height;
		auto focal_length = 1.0f;

		origin = make_float3(0, 0, 0);
		horizontal = make_float3(viewport_width, 0, 0);
		vertical = make_float3(0, viewport_height, 0);
		lower_left_corner = origin - horizontal / 2 - vertical / 2 - make_float3(0, 0, focal_length);
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
