#pragma once

struct Params
{
    uchar4* image;
};

using vec3 = float3;
using color = vec3;

__forceinline__ __device__ void write_color(uchar4& out, const color& pixel_color)
{
    auto r = pixel_color.x;
    auto g = pixel_color.y;
    auto b = pixel_color.z;
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

__forceinline__ __device__ float3 unit_vector(const float3& v)
{
  float invLen = 1.0f / sqrtf(dot(v, v));
  return v * invLen;
}