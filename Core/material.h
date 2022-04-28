#pragma once

#include "vec_math.h"
#include <glm/glm.hpp>

class Material
{
public:
	enum Type
	{
		Lambertian = 0,
		Metal,
		Dielectric
	};

	inline const Type GetType() const { return m_type; }
	inline const float3& GetAlbedo() const { return make_float3(m_albedo.r, m_albedo.g, m_albedo.b); }
	inline const float GetFuzz() const { return m_fuzz; }
	inline const float GetIR() const { return m_ir; }

private:
	Type m_type;
	glm::vec3 m_albedo;
	float m_fuzz;
	float m_ir;
};



