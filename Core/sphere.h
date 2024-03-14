#pragma once

#include "material.h"

class Sphere
{
public:
	Sphere(const glm::vec3& center, float radius, const Material& material)
		: m_center(center), m_radius(radius), m_material(material)
	{
	}

	inline glm::vec3 GetCenter() const { return m_center; }
	inline const float GetRadius() const { return m_radius; }
	inline const Material& GetMaterial() const { return m_material; }

	//inline glm::aabb GetAABB() const
	//{
	//	float radius = fabsf(m_radius);
	//	return {
	//		m_center.x - m_radius,
	//		m_center.y - m_radius,
	//		m_center.z - m_radius,
	//		m_center.x + m_radius,
	//		m_center.y + m_radius,
	//		m_center.z + m_radius
	//	};
	//}
	inline float GetMinX() const { return m_center.x - m_radius; }
	inline float GetMinY() const { return m_center.y - m_radius; }
	inline float GetMinZ() const { return m_center.z - m_radius; }
	inline float GetMaxX() const { return m_center.x + m_radius; }
	inline float GetMaxY() const { return m_center.y + m_radius; }
	inline float GetMaxZ() const { return m_center.z + m_radius; }

	inline void SetMaterial(const Material& material) { m_material = material; }

private:
	glm::vec3 m_center;
	float m_radius;
	Material m_material;
};
