#pragma once

#include "rtweekend.h"

class camera
{
public:
	camera(const glm::vec3& lookfrom, const glm::vec3& lookat, const glm::vec3& vup, float vfov, float aspect_ratio, float aperture, float focus_dist)
	{
		auto theta = degrees_to_radians(vfov);
		auto h = tanf(theta / 2);
		viewport_height = 2.0f * h;
		viewport_width = aspect_ratio * viewport_height;
		this->focus_dist = focus_dist;
		

		origin = lookfrom;
		target = lookat;
		world_up = vup;
		update_uvw();

		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist * w;
		lens_radius = aperture / 2;
	}

	void set_sbt(RayGenSbtRecord& rg_sbt)
	{
		rg_sbt.data.origin = make_float3(origin);
		rg_sbt.data.horizontal = make_float3(horizontal);
		rg_sbt.data.vertical = make_float3(vertical);
		rg_sbt.data.lower_left_corner = make_float3(lower_left_corner);
		rg_sbt.data.u = make_float3(u);
		rg_sbt.data.v = make_float3(v);
		rg_sbt.data.w = make_float3(w);
		rg_sbt.data.lens_radius = lens_radius;
	}

	void update_uvw()
	{
		w = normalize(origin - target);
		u = normalize(cross(world_up, w));
		v = cross(w, u);

		horizontal = focus_dist * viewport_width * u;
		vertical = focus_dist * viewport_height * v;
		lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist * w;
	}

	void move_forward(float speed)
	{
		origin += -w * speed;
		update_uvw();
	}

	void move_right(float speed)
	{
		origin += u * speed;
		update_uvw();
	}

	void move_up(float speed)
	{
		origin += v * speed;
		update_uvw();
	}

	void pitch(float speed)
	{
		glm::quat front = glm::quatLookAt(-w, world_up);
		front = glm::rotate(front, glm::radians(speed), glm::vec3(1, 0, 0));
		target = origin + normalize(-w * front);
		update_uvw();
	}

	void yaw(float speed)
	{
		glm::quat front = glm::quatLookAt(-w, world_up);
		front = glm::rotate(front, glm::radians(speed), glm::vec3(0, 1, 0));
		target = origin + normalize(-w * front);
		update_uvw();
	}

	void roll(float speed)
	{
		glm::quat front = glm::quatLookAt(-w, world_up);
		front = glm::rotate(front, glm::radians(speed), glm::vec3(0, 0, 1));
		target = origin + normalize(-w * front);
		update_uvw();
	}

private:
	float viewport_height;
	float viewport_width;
	float focus_dist;

	glm::vec3 origin;
	glm::vec3 target;
	glm::vec3 world_up;
	glm::vec3 euler_angles;
	glm::vec3 lower_left_corner;
	glm::vec3 horizontal;
	glm::vec3 vertical;
	glm::vec3 u, v, w;
	float lens_radius;
};
