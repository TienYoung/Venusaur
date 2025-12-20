#include <optix.h>

#include "vec_math.h"
#include "random.h"

#include "RayTracer.h"

bool __forceinline__ __device__ near_zero(const float3& e)
{
	// Return true if the vector is close to zero in all dimensions.
	const auto s = 1e-8f;
	return (fabs(e.x) < s) && (fabs(e.y) < s) && (fabs(e.z) < s);
}


__forceinline__ __device__ float3 toSRGB(const float3& c)
{
	float  invGamma = 1.0f / 2.4f;
	float3 powed = make_float3(powf(c.x, invGamma), powf(c.y, invGamma), powf(c.z, invGamma));
	return make_float3(
		c.x < 0.0031308f ? 12.92f * c.x : 1.055f * powed.x - 0.055f,
		c.y < 0.0031308f ? 12.92f * c.y : 1.055f * powed.y - 0.055f,
		c.z < 0.0031308f ? 12.92f * c.z : 1.055f * powed.z - 0.055f);
}

//__forceinline__ __device__ float dequantizeUnsigned8Bits( const unsigned char i )
//{
//    enum { N = (1 << 8) - 1 };
//    return min((float)i / (float)N), 1.f)
//}
__forceinline__ __device__ unsigned char quantizeUnsigned8Bits(float x)
{
	x = clamp(x, 0.0f, 1.0f);
	enum { N = (1 << 8) - 1, Np1 = (1 << 8) };
	return (unsigned char)min((unsigned int)(x * (float)Np1), (unsigned int)N);
}

__forceinline__ __device__ uchar4 make_color(const float3& c)
{
	// first apply gamma, then convert to unsigned char
	//float3 srgb = toSRGB(clamp(c, 0.0f, 1.0f));
	float3 srgb = clamp(c, 0.0f, 1.0f);
	return make_uchar4(quantizeUnsigned8Bits(srgb.x), quantizeUnsigned8Bits(srgb.y), quantizeUnsigned8Bits(srgb.z), 255u);
}
__forceinline__ __device__ uchar4 make_color(const float4& c)
{
	return make_color(make_float3(c.x, c.y, c.z));
}

extern "C" 
{
    __constant__ Params params;
}

struct PRD
{
	float3 attenuation;
	float3 origin;
	float3 direction;
	unsigned int seed;
	int done;
	int depth;
};

static __forceinline__ __device__ void* unpackPointer(unsigned int i0, unsigned int i1)
{
	const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
	void* ptr = reinterpret_cast<void*>(uptr);
	return ptr;
}


static __forceinline__ __device__ void  packPointer(void* ptr, unsigned int& i0, unsigned int& i1)
{
	const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}


static __forceinline__ __device__ PRD* getPRD()
{
	const unsigned int u0 = optixGetPayload_0();
	const unsigned int u1 = optixGetPayload_1();
	return reinterpret_cast<PRD*>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ float random_float(unsigned int& seed)
{
	// Returns a random real in [0,1).
	return rnd(seed);
}

static __forceinline__ __device__ float random_float(unsigned int& seed, float min, float max)
{
	// Returns a random real in [min,max).
	return min + (max - min) * random_float(seed);
}

static __forceinline__ __device__ float3 random(unsigned int& seed)
{
	return make_float3(
		random_float(seed),
		random_float(seed),
		random_float(seed)
	);
}

static __forceinline__ __device__ float3 random(unsigned int& seed, float min, float max)
{
	return make_float3(
		random_float(seed, min, max),
		random_float(seed, min, max),
		random_float(seed, min, max)
	);
}

static __forceinline__ __device__ float3 random_in_unit_sphere(unsigned int& seed) 
{
	while (true)
	{
		auto p = random(seed, -1, 1);
		if (dot(p, p) >= 1) continue;
		return p;
	}
}

static __forceinline__ __device__ float3 random_unit_vector(unsigned int& seed)
{
	return normalize(random_in_unit_sphere(seed));
}

static __forceinline__ __device__ float3 random_in_hemisphere(unsigned int& seed, const float3& normal)
{
	float3 in_unit_sphere = random_in_unit_sphere(seed);
	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

static __forceinline__ __device__ float3 random_in_unit_disk(unsigned int& seed)
{
	while (true) 
	{
		auto p = make_float3(random_float(seed, -1, 1), random_float(seed, -1, 1), 0);
		if (dot(p, p) >= 1) continue;
		return p;
	}
}

static __forceinline__ __device__ void get_ray(float s, float t, float3& origin, float3& direction, unsigned int& seed)
{
	RayGenData* rtData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
	float3 rd = params.lens_radius * random_in_unit_disk(seed);
	float3 u = normalize(params.u);
	float3 v = normalize(params.v);
	float3 offset = u * rd.x + v * rd.y;

	origin = params.origin + offset;
	direction = params.w + s * params.u * 0.5 + t * params.v * 0.5 - offset;
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_index = optixGetLaunchIndex();
	const unsigned int image_index = launch_index.y * params.width + launch_index.x;;

	float3 pixel_color = make_float3(0.0f);
	unsigned int seed = tea<4>(image_index, params.subframe_index);
	for (int s = 0; s < params.samples_per_pixel; ++s)
	{
		const int max_depth = 4;
		auto u = 2 * float(launch_index.x + random_float(seed)) / (params.width - 1) - 1;
		auto v = 2 * float(launch_index.y + random_float(seed)) / (params.height - 1) - 1;
		float3 origin;
		float3 direction;
		get_ray(u, v, origin, direction, seed);

		PRD prd;
		prd.attenuation = make_float3(1.0f);
		prd.origin = origin;
		prd.direction = direction;
		prd.seed = seed;
		prd.depth = max_depth - 1;

		unsigned int p0, p1;
		packPointer(&prd, p0, p1);

		// Trace the ray against our scene hierarchy
		optixTrace(
			params.handle,
			prd.origin,
			prd.direction,
			0.001f,                // Min intersection distance
			1e16f,               // Max intersection distance
			0.0f,                // rayTime -- used for motion blur
			OptixVisibilityMask(255), // Specify always visible
			OPTIX_RAY_FLAG_NONE,
			0,                   // SBT offset   -- See SBT discussion
			1,                   // SBT stride   -- See SBT discussion
			0,                   // missSBTIndex -- See SBT discussion
			p0, p1);
		pixel_color += prd.attenuation;
	}

	float3 accum_color = pixel_color / static_cast<float>(params.samples_per_pixel);

	if (params.subframe_index > 0)
	{
		const float                 a = 1.0f / static_cast<float>(params.subframe_index + 1);
		const float3 accum_color_prev = make_float3(params.accum[image_index]);
		accum_color = lerp(accum_color_prev, accum_color, a);
	}

	params.accum[image_index] = make_float4(accum_color, 1.0f);
	params.image[image_index] = make_color(accum_color);
}

static __forceinline__ __device__ bool set_face_normal(const float3& direction, float3& outward_normal)
{
	bool front_face = dot(direction, outward_normal) < 0;
	outward_normal = front_face ? outward_normal : -outward_normal;
	return front_face;
}

#define OPTIX_HIT_KIND_CUSTOM_SPHERE_FRONT_FACE 0x00
#define OPTIX_HIT_KIND_CUSTOM_SPHERE_BACK_FACE 0x01

//extern "C" __global__ void __intersection__hit_sphere()
//{
//	SphereHitGroupData* rtData = reinterpret_cast<SphereHitGroupData*>(optixGetSbtDataPointer());
//	const int    prim_idx = optixGetPrimitiveIndex();
//
//	float3 origin = optixGetWorldRayOrigin();
//	float3 direction = optixGetWorldRayDirection();
//	float t_min = optixGetRayTmin();
//	float t_max = optixGetRayTmax();
//
//	float3 oc = origin - rtData->center;
//	auto a = dot(direction, direction);
//	auto half_b = dot(oc, direction);
//	auto c = dot(oc, oc) - rtData->radius * rtData->radius;
//
//	auto discriminant = half_b * half_b - a * c;
//	if (discriminant < 0) return;
//	auto sqrtd = sqrt(discriminant);
//
//	auto root = (-half_b - sqrtd) / a;
//	if (root < t_min || t_max < root)
//	{
//		root = (-half_b + sqrtd) / a;
//		if (root < t_min || t_max < root) return;
//	}
//
//	float t = root;
//	float3 p = (origin + direction * root);
//	float3 normal = (p - rtData->center) / rtData->radius;
//	bool front_face = set_face_normal(direction, normal);
//
//	optixReportIntersection(
//		t,
//		front_face ? OPTIX_HIT_KIND_CUSTOM_SPHERE_FRONT_FACE : OPTIX_HIT_KIND_CUSTOM_SPHERE_BACK_FACE,
//		__float_as_uint(p.x),
//		__float_as_uint(p.y),
//		__float_as_uint(p.z),
//		__float_as_uint(normal.x),
//		__float_as_uint(normal.y),
//		__float_as_uint(normal.z)
//	);
//}

extern "C" __global__ void __closesthit__lambertian()
{
	PRD* prd = getPRD();
	if (prd->depth > 0)
	{
		float3 p = make_float3(
			__uint_as_float(optixGetAttribute_0()),
			__uint_as_float(optixGetAttribute_1()),
			__uint_as_float(optixGetAttribute_2())
		);
		float3 normal = make_float3(
			__uint_as_float(optixGetAttribute_3()),
			__uint_as_float(optixGetAttribute_4()),
			__uint_as_float(optixGetAttribute_5())
		);

		auto scatter_direction = normal + random_unit_vector(prd->seed);
		// Catch degenerate scatter direction
		if (near_zero(scatter_direction))
			scatter_direction = normal;

		prd->origin = p;
		prd->direction = scatter_direction;
		prd->depth -= 1;

		unsigned int p0, p1;
		packPointer(prd, p0, p1);
		optixTrace(
			params.handle,
			prd->origin,
			prd->direction,
			0.001f,                // Min intersection distance
			1e16f,               // Max intersection distance
			0.0f,                // rayTime -- used for motion blur
			OptixVisibilityMask(255), // Specify always visible
			OPTIX_RAY_FLAG_NONE,
			0,                   // SBT offset   -- See SBT discussion
			1,                   // SBT stride   -- See SBT discussion
			0,                   // missSBTIndex -- See SBT discussion
			p0, p1);
		HitGroupData* rtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
		MaterialData mat = rtData->material;
		prd->attenuation *= mat.albedo;
	}
	else
	{
		prd->attenuation = make_float3(0.0, 0.0, 0.0);
	}
}

extern "C" __global__ void __closesthit__metal()
{
	PRD* prd = getPRD();
	if (prd->depth > 0)
	{
		float3 p = make_float3(
			__uint_as_float(optixGetAttribute_0()),
			__uint_as_float(optixGetAttribute_1()),
			__uint_as_float(optixGetAttribute_2())
		);
		float3 normal = make_float3(
			__uint_as_float(optixGetAttribute_3()),
			__uint_as_float(optixGetAttribute_4()),
			__uint_as_float(optixGetAttribute_5())
		);

		HitGroupData* rtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
		MaterialData mat = rtData->material;
		float3 reflected = reflect(normalize(prd->direction), normal);
		prd->origin = p;
		prd->direction = reflected + mat.fuzz * random_in_unit_sphere(prd->seed);
		if (dot(prd->direction, normal) > 0)
		{
			prd->depth -= 1;

			unsigned int p0, p1;
			packPointer(prd, p0, p1);
			optixTrace(
				params.handle,
				prd->origin,
				prd->direction,
				0.001f,                // Min intersection distance
				1e16f,               // Max intersection distance
				0.0f,                // rayTime -- used for motion blur
				OptixVisibilityMask(255), // Specify always visible
				OPTIX_RAY_FLAG_NONE,
				0,                   // SBT offset   -- See SBT discussion
				1,                   // SBT stride   -- See SBT discussion
				0,                   // missSBTIndex -- See SBT discussion
				p0, p1);
			prd->attenuation *= mat.albedo;
		}
		else
		{
			prd->attenuation = make_float3(0.0, 0.0, 0.0);
		}
	}
	else
	{
		prd->attenuation = make_float3(0.0, 0.0, 0.0);
	}
}

static __forceinline__ __device__ float reflectance(float cosine, float ref_idx)
{
	// Use Schlick's approximation for reflectance.
	auto r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * __powf((1 - cosine), 5);
}

extern "C" __global__ void __closesthit__dielectric()
{
	PRD* prd = getPRD();
	if (prd->depth > 0)
	{
		float3 p = make_float3(
			__uint_as_float(optixGetAttribute_0()),
			__uint_as_float(optixGetAttribute_1()),
			__uint_as_float(optixGetAttribute_2())
		);
		float3 normal = make_float3(
			__uint_as_float(optixGetAttribute_3()),
			__uint_as_float(optixGetAttribute_4()),
			__uint_as_float(optixGetAttribute_5())
		);

		HitGroupData* rtData = reinterpret_cast<HitGroupData*>(optixGetSbtDataPointer());
		MaterialData mat = rtData->material;
		float refraction_ratio = mat.ir;
		if (optixGetHitKind() == OPTIX_HIT_KIND_CUSTOM_SPHERE_FRONT_FACE)
		{
			refraction_ratio = (1.0f / mat.ir);
		}

		float3 unit_direction = normalize(prd->direction);
		float cos_theta = fminf(dot(-unit_direction, normal), 1.0f);
		float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
		float3 direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(prd->seed))
			direction = reflect(unit_direction, normal);
		else
			direction = refract(unit_direction, normal, refraction_ratio);

		prd->origin = p;
		prd->direction = direction;
		prd->depth -= 1;

		unsigned int p0, p1;
		packPointer(prd, p0, p1);
		optixTrace(
			params.handle,
			prd->origin,
			prd->direction,
			0.001f,                // Min intersection distance
			1e16f,               // Max intersection distance
			0.0f,                // rayTime -- used for motion blur
			OptixVisibilityMask(255), // Specify always visible
			OPTIX_RAY_FLAG_NONE,
			0,                   // SBT offset   -- See SBT discussion
			1,                   // SBT stride   -- See SBT discussion
			0,                   // missSBTIndex -- See SBT discussion
			p0, p1);
	}
	else
	{
		prd->attenuation = make_float3(0.0, 0.0, 0.0);
	}
}

extern "C" __global__ void __miss__ms()
{
    float3 unit_direction = normalize(optixGetWorldRayDirection());
    auto t = 0.5f * (unit_direction.y + 1.0f);

	PRD* prd = getPRD();
    prd->attenuation = lerp(make_float3(1.0f), make_float3(0.5f, 0.7f, 1.0f), t);
	prd->depth -= 1;
}