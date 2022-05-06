#pragma once

#include <optix.h>

#include <vector>
#include <random>

#include "Sphere.h"

class Scene
{
public:
	Scene()
	{
		Material ground_material(Material::Lambertian, glm::vec3(0.5f, 0.5f, 0.5f));
		Sphere ground_spherer(glm::vec3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_material);
		uint32_t index = 0;
		m_spheres.push_back(ground_spherer);
		m_aabbs.push_back(ground_spherer.GetAABB());
		m_indices.push_back(index++);

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				auto choose_mat = random_float();
				glm::vec3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

				if (glm::length(center - glm::vec3(4, 0.2, 0)) > 0.9) {
					Material sphere_material(Material::Lambertian);
					Sphere sphere(center, 0.2f, sphere_material);

					if (choose_mat < 0.8) {
						// diffuse
						auto albedo = random() * random();
						sphere_material = Material(Material::Lambertian, albedo);
						sphere.SetMaterial(sphere_material);
						m_spheres.push_back(sphere);
						m_aabbs.push_back(sphere.GetAABB());
						m_indices.push_back(index++);
					}
					else if (choose_mat < 0.95) {
						// metal
						auto albedo = random(0.5, 1);
						auto fuzz = random_float(0, 0.5);
						sphere_material = Material(Material::Metal, albedo, fuzz);
						sphere.SetMaterial(sphere_material);
						m_spheres.push_back(sphere);
						m_aabbs.push_back(sphere.GetAABB());
						m_indices.push_back(index++);
					}
					else {
						// glass
						sphere_material = Material(Material::Dielectric, glm::vec3(0.0f), 0.0f, 1.5);
						sphere.SetMaterial(sphere_material);
						m_spheres.push_back(sphere);
						m_aabbs.push_back(sphere.GetAABB());
						m_indices.push_back(index++);
					}
				}
			}
		}

		auto material1 = Material(Material::Dielectric, glm::vec3(0.0f), 0.0f, 1.5);
		Sphere sphere(glm::vec3(0, 1, 0), 1.0, material1);
		m_spheres.push_back(sphere);
		m_aabbs.push_back(sphere.GetAABB());
		m_indices.push_back(index++);

		auto material2 = Material(Material::Lambertian, glm::vec3(0.4, 0.2, 0.1));
		sphere = Sphere(glm::vec3(-4, 1, 0), 1.0, material2);
		m_spheres.push_back(sphere);
		m_aabbs.push_back(sphere.GetAABB());
		m_indices.push_back(index++);

		auto material3 = Material(Material::Metal, glm::vec3(0.7, 0.6, 0.5), 0.0);
		sphere = Sphere(glm::vec3(4, 1, 0), 1.0, material3);
		m_spheres.push_back(sphere);
		m_aabbs.push_back(sphere.GetAABB());
		m_indices.push_back(index++);

	}

	std::vector<Sphere> m_spheres;
	std::vector<OptixAabb> m_aabbs;
	std::vector<uint32_t> m_indices;

private:
	static float random_float() {
		static std::uniform_real_distribution<float> distribution(0.0, 1.0);
		static std::mt19937 generator;
		return distribution(generator);
	}

	static float random_float(float min, float max) {
		// Returns a random real in [min,max).
		return min + (max - min) * random_float();
	}

	static glm::vec3 random_in_unit_disk() {
		while (true) {
			auto p = glm::vec3(random_float(-1, 1), random_float(-1, 1), 0);
			if (p.length() >= 1) continue;
			return p;
		}
	}

	static glm::vec3 random() {
		return glm::vec3(random_float(), random_float(), random_float());
	}

	inline static glm::vec3 random(float min, float max) {
		return glm::vec3(random_float(min, max), random_float(min, max), random_float(min, max));
	}
};