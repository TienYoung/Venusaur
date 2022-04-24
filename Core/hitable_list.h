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
		aabbs.push_back(genAABB(object));
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