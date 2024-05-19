#pragma once

#include <vector>
#include <random>

#include "vec_math.h"

class Scene
{
public:
	enum MaterialType
	{
		Lambertian = 0,
		Metal,
		Dielectric
	};

	struct Object
	{
		size_t sphere;
		size_t material;
		MaterialType type;
	};

public:
	Scene()
	{
		uint32_t index = 0;
		m_materials.emplace_back(MaterialData{ 0.5f, 0.5f, 0.5f });
		m_spheres.emplace_back(SphereData{ 0.0f, -1000.0f, 0.0f, 1000.0f });
		m_objects.emplace_back(Object{ index, index, MaterialType::Lambertian });
		index++;

		for (int a = -11; a < 11; a++) {
			for (int b = -11; b < 11; b++) {
				auto choose_mat = random_float();
				float3 center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

				if (length(center - float3(4.0f, 0.2f, 0.0f)) > 0.9f) {
					m_spheres.emplace_back(SphereData{ center, 0.2f });

					if (choose_mat < 0.8) {
						// diffuse
						auto albedo = random() * random();
						m_materials.emplace_back(MaterialData{ albedo });
						m_objects.emplace_back(Object{ index, index, MaterialType::Lambertian });
					}
					else if (choose_mat < 0.95) {
						// metal
						auto albedo = random(0.5, 1);
						auto fuzz = random_float(0, 0.5);
						m_materials.emplace_back(MaterialData{ albedo, fuzz });
						m_objects.emplace_back(Object{ index, index, MaterialType::Metal });
					}
					else {
						// glass
						m_materials.emplace_back(MaterialData{ .ir = 1.5f });
						m_objects.emplace_back(Object{ index, index, MaterialType::Dielectric });
					}
					index++;
				}
			}
		}

		m_spheres.emplace_back(SphereData{ 0.0f, 1.0f, 0.0f, 1.0f });
		m_materials.emplace_back(MaterialData{ .ir = 1.5f });
		m_objects.emplace_back(Object{ index, index, MaterialType::Dielectric });
		index++;

		m_spheres.emplace_back(SphereData{ -4.0f, 1.0f, 0.0f, 1.0f });
		m_materials.emplace_back(MaterialData{ 0.4f, 0.2f, 0.1f });
		m_objects.emplace_back(Object{ index, index, MaterialType::Lambertian });
		index++;

		m_spheres.emplace_back(SphereData{ 4.0f, 1.0f, 0.0f, 1.0f });
		m_materials.emplace_back(MaterialData{ 0.7f, 0.6f, 0.5f });
		m_objects.emplace_back(Object{ index, index, MaterialType::Metal });
		index++;
	}

	size_t getSpheresData(const float3*& centers, const float*& radii)
	{
		centers = reinterpret_cast<const float3*>(m_spheres.data());
		radii = reinterpret_cast<const float*>(centers + 3);
		return m_spheres.size();
	}

	MaterialData getObjectMaterial(size_t index)
	{
		size_t i = m_objects[index].material;
		return m_materials[i];
	}

	const std::vector<Object>& getObjectsRef()
	{
		return m_objects;
	}


private:
	std::vector<SphereData> m_spheres;
	std::vector<MaterialData> m_materials;
	std::vector<Object> m_objects;

	static float random_float() {
		static std::uniform_real_distribution<float> distribution(0.0, 1.0);
		static std::mt19937 generator;
		return distribution(generator);
	}

	static float random_float(float min, float max) {
		// Returns a random real in [min,max).
		return min + (max - min) * random_float();
	}

	static float3 random_in_unit_disk() {
		while (true) {
			auto p = float3(random_float(-1, 1), random_float(-1, 1), 0);
			if (length(p) >= 1) continue;
			return p;
		}
	}

	static float3 random() {
		return float3(random_float(), random_float(), random_float());
	}

	inline static float3 random(float min, float max) {
		return float3(random_float(min, max), random_float(min, max), random_float(min, max));
	}
};