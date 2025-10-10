#pragma once

#include "sphere.h"

struct ParamsAntialiasing : public ParamsSphere
{
    unsigned int samples_per_pixel;
};