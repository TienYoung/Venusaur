#pragma once

#include "vec_math.h"

#include "material.h"

#include <random>


#include <glm/glm.hpp>
#include <glm/ext.hpp>


inline float degrees_to_radians(float degrees) {
	return degrees * M_PIf / 180.0f;
}

inline float random_float() 
{
	static std::uniform_real_distribution<float> distribution(0.0, 1.0);
	static std::mt19937 generator;
	return distribution(generator);
}

inline float random_float(float min, float max) 
{
	return min + (max - min) * random_float();
}

inline static float3 random_float3()
{
	return make_float3(random_float(), random_float(), random_float());
}

inline static float3 random_float3(float min, float max)
{
	return min + (max - min) * make_float3(random_float(), random_float(), random_float());
}

/** Convert from glm
* @{
*/
inline static __host__ float3 make_float3(const glm::vec3& v0) { return make_float3(v0.x, v0.y, v0.z); }
inline static __host__ float4 make_float4(const glm::vec4& v0) { return make_float4(v0.x, v0.y, v0.z, v0.w); }
/** @} */