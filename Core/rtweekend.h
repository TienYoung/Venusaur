#pragma once

#include "vec_math.h"

#include "material.h"

#include <random>

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