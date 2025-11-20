#pragma once

#include <params.h>

struct DiffuseParams
{
    uchar4* image;
    float3 camera_center;
    float3 pixel00_loc;
    float3 pixel_delta_u;
    float3 pixel_delta_v;
    unsigned int samples_per_pixel;
    unsigned int subframe_index;
    OptixTraversableHandle handle;
};

struct DiffusePayload
{
    unsigned int seed;
    unsigned int depth;
    float3 origin;
    float3 direction;
    float3 diffuse;
};

struct Onb
{
    __forceinline__ __device__ Onb(const float3& normal)
    {
        m_normal = normal;

        if( fabs(m_normal.x) > fabs(m_normal.z) )
        {
        m_binormal.x = -m_normal.y;
        m_binormal.y =  m_normal.x;
        m_binormal.z =  0;
        }
        else
        {
        m_binormal.x =  0;
        m_binormal.y = -m_normal.z;
        m_binormal.z =  m_normal.y;
        }

        m_binormal = unit_vector(m_binormal);
        m_tangent = cross( m_binormal, m_normal );
    }

    __forceinline__ __device__ void inverse_transform(float3& p) const
    {
        p = p.x*m_tangent + p.y*m_binormal + p.z*m_normal;
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};