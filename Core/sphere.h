#pragma once

class Sphere
{
public:
	Sphere(const glm::vec3& center, float radius, const Material& material)
		: m_center(center), m_radius(radius), m_material(material)
	{
	}

	inline const float3& GetCenter() const { return make_float3(m_center.x, m_center.y, m_center.z); }
	inline const float GetRadius() const { return m_radius; }
	inline const Material& GetMaterial() const { return m_material; }

	inline const OptixAabb& GetAABB() const
	{
		float radius = fabsf(m_radius);
		return {
			m_center.x - m_radius,
			m_center.y - m_radius,
			m_center.z - m_radius,
			m_center.x + m_radius,
			m_center.y + m_radius,
			m_center.z + m_radius
		};
	}
private:
	glm::vec3 m_center;
	float m_radius;
	Material m_material;
};
