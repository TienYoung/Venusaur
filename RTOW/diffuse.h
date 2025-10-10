#pragma once

#include "antialiasing.h"

struct ParamsDiffuse: public ParamsAntialiasing
{
    unsigned int subframe_index;
};

struct DiffusePayload
{
    unsigned int seed;
    unsigned int depth;
    float3 diffuse;
};
