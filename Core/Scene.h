#pragma once

#include <optix.h>

#include <vector>

#include "Sphere.h"

class Scene
{
public:
	Scene();
	~Scene();

	std::vector<Sphere> m_spheres;
	std::vector<OptixAabb> m_aabbs;
	std::vector<uint32_t> m_indices;
};