#pragma once

#include <vector>

class hittable_list
{
public:
	hittable_list() {}
	~hittable_list() 
	{
		clear();
	}

	void clear() 
	{
		objects.clear(); 
		hitgroup_keys.clear();
		aabbs.clear();
		indices.clear();
	}
	void add(const SphereHitGroupData& object, const char* hitgroup_name)
	{
		objects.push_back(HitGroupSbtRecord{ {},object });
		hitgroup_keys.push_back(hitgroup_name);
		aabbs.push_back(gen_aabb(object));
		indices.push_back(static_cast<uint32_t>(indices.size()));
	}

	const size_t aabb_buffer(const OptixAabb* & buffer) const
	{
		buffer = aabbs.data();
		return aabbs.size() * sizeof(OptixAabb);
	}

	const size_t indices_buffer(const uint32_t* & buffer) const
	{
		buffer = indices.data();
		return indices.size() * sizeof(uint32_t);
	}

	unsigned int objects_count() const
	{
		return static_cast<unsigned int>(objects.size());
	}

	const HitGroupSbtRecord* objects_stb() const 
	{
		return objects.data();
	}

	HitGroupSbtRecord* get_hitgroup_stb(size_t index)
	{
		return &objects[index];
	}

	const char* get_material_hitgroup(size_t index) const
	{
		return hitgroup_keys[index];
	}

private:
	std::vector<HitGroupSbtRecord> objects;
	std::vector<const char*> hitgroup_keys;
	std::vector<OptixAabb> aabbs;
	std::vector<uint32_t> indices;
};

hittable_list random_scene()
{
	hittable_list world;

	auto ground_material = make_lambertian(make_float3(0.5, 0.5, 0.5));
	world.add(make_sphere(make_float3(0, -1000, 0), 1000, ground_material), "lambertian");

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_float();
			float3 center = make_float3(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

			if (length(center - make_float3(4, 0.2, 0)) > 0.9) {
				material sphere_material;

				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = random_float3() * random_float3();
					sphere_material = make_lambertian(albedo);
					world.add(make_sphere(center, 0.2, sphere_material), "lambertian");
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = random_float3(0.5, 1);
					auto fuzz = random_float(0, 0.5);
					sphere_material = make_metal(albedo, fuzz);
					world.add(make_sphere(center, 0.2, sphere_material), "metal");
				}
				else {
					// glass
					sphere_material = make_dielectric(1.5);
					world.add(make_sphere(center, 0.2, sphere_material), "dielectric");
				}
			}
		}
	}

	auto material1 = make_dielectric(1.5);
	world.add(make_sphere(make_float3(0, 1, 0), 1.0, material1), "dielectric");

	auto material2 = make_lambertian(make_float3(0.4, 0.2, 0.1));
	world.add(make_sphere(make_float3(-4, 1, 0), 1.0, material2), "lambertian");

	auto material3 = make_metal(make_float3(0.7, 0.6, 0.5), 0.0);
	world.add(make_sphere(make_float3(4, 1, 0), 1.0, material3), "metal");

	return world;
}