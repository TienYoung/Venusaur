#pragma once

#include "vec_math.h"

#include "material.h"

inline float degrees_to_radians(float degrees) {
	return degrees * M_PIf / 180.0f;
}