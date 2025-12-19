#pragma once

__forceinline__ __device__ float linear_to_gamma(float linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}

__forceinline__ __device__ void write_color(uchar4& out, const float3& pixel_color)
{
    auto r = pixel_color.x;
    auto g = pixel_color.y;
    auto b = pixel_color.z;

    // Apply a linear to gamma transform for gamma 2
    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    // Translate the [0,1] component values to the byte range [0,255].
    int rbyte = int(255.999f * r);
    int gbyte = int(255.999f * g);
    int bbyte = int(255.999f * b);
    // Write out the pixel color components.
    out.x = static_cast<unsigned char>(rbyte);
    out.y = static_cast<unsigned char>(gbyte); 
    out.z = static_cast<unsigned char>(bbyte);
    out.w = 255; // Alpha channel
}

__forceinline__ __device__ float3 operator+(const float3& a, const float3& b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__forceinline__ __device__ float3 operator+(const float3& a, const float b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}
__forceinline__ __device__ float3 operator+(const float a, const float3& b)
{
  return make_float3(a + b.x, a + b.y, a + b.z);
}
__forceinline__ __device__ void operator+=(float3& a, const float3& b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}

__forceinline__ __device__ float3 operator-(const float3& a, const float3& b)
{
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__forceinline__ __device__ float3 operator-(const float3& a, const float b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}
__forceinline__ __device__ float3 operator-(const float a, const float3& b)
{
  return make_float3(a - b.x, a - b.y, a - b.z);
}
__forceinline__ __device__ void operator-=(float3& a, const float3& b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

__forceinline__ __device__ float3 operator*(const float3& a, const float3& b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__forceinline__ __device__ float3 operator*(const float3& a, const float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__forceinline__ __device__ float3 operator*(const float s, const float3& a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}
__forceinline__ __device__ void operator*=(float3& a, const float3& s)
{
  a.x *= s.x; a.y *= s.y; a.z *= s.z;
}
__forceinline__ __device__ void operator*=(float3& a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s;
}

__forceinline__ __device__  float3 operator/(const float3& a, const float3& b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__forceinline__ __device__  float3 operator/(const float3& a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
__forceinline__ __device__  float3 operator/(const float s, const float3& a)
{
  return make_float3( s/a.x, s/a.y, s/a.z );
}
__forceinline__ __device__  void operator/=(float3& a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}

__forceinline__ __device__ float dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__forceinline__ __device__ float3 cross(const float3& a, const float3& b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__forceinline__ __device__ float3 unit_vector(const float3& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}

__forceinline__ __device__  float3 ray_color(const float3& ray_direction) {
    float3 unit_direction = unit_vector(ray_direction);
    auto a = 0.5f * (unit_direction.y + 1.0f);
    return (1.0f - a) * float3(1.0f, 1.0f, 1.0f) + a * float3(0.5f, 0.7f, 1.0f);
}

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