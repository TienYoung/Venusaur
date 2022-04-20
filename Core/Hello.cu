#include <optix.h>

#include "random.cuh"

#include "Hello.h"

extern "C" 
{
    __constant__ Params params;
}

struct PRD
{
	float3 color;
	float3 origin;
	float3 direction;
	unsigned int seed;
	int hitted;
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

//static __forceinline__ __device__ float3 random(unsigned int& seed)
//{
//	return make_float3(
//		random_float(seed),
//		random_float(seed),
//		random_float(seed)
//	);
//}

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
	while (true) {
		auto p = random(seed, -1, 1);
		if (dot(p, p) >= 1) continue;
		return p;
	}
}

static __forceinline__ __device__ float3 random_unit_vector(unsigned int& seed)
{
	return normalize(random_in_unit_sphere(seed));
}

static __forceinline__ __device__ float3 random_in_hemisphere(unsigned int& seed, const float3& normal) {
	float3 in_unit_sphere = random_in_unit_sphere(seed);
	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

static __forceinline__ __device__ void get_ray(float u, float v, float3& origin, float3& direction)
{
	RayGenData* rtData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());
	origin = rtData->origin;
	direction = rtData->lower_left_corner + u * rtData->horizontal + v * rtData->vertical - rtData->origin;
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_index = optixGetLaunchIndex();
	const unsigned int image_index = launch_index.y * params.image_width + launch_index.x;;

	float3 pixel_color = make_float3(0.0f);
	unsigned int seed = tea<4>(image_index, 0);
	for (int s = 0; s < params.samples_per_pixel; ++s)
	{
		const int max_depth = 50;
		auto u = double(launch_index.x + random_float(seed)) / (params.image_width - 1);
		auto v = double(launch_index.y + random_float(seed)) / (params.image_height - 1);
		float3 origin;
		float3 direction;
		get_ray(u, v, origin, direction);

		PRD prd;
		prd.color = make_float3(0.0f);
		prd.origin = origin;
		prd.direction = direction;
		prd.seed = seed;
		prd.hitted = false;
		prd.depth = max_depth;
		
		unsigned int p0, p1;
		packPointer(&prd, p0, p1);
		do
		{
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
		} while (prd.depth > 0 && prd.hitted);
		prd.color *= pow(0.5, max_depth - prd.depth - 1);
		pixel_color += prd.color;
	}

    params.image[launch_index.y * params.image_width + launch_index.x] = make_float4(pixel_color / params.samples_per_pixel, 1.0f);
}

static __forceinline__ __device__ bool set_face_normal(const float3& direction, float3& outward_normal)
{
	bool front_face = dot(direction, outward_normal) < 0;
	outward_normal = front_face ? outward_normal : -outward_normal;
}

extern "C" __global__ void __intersection__hit_sphere()
{
	SphereHitGroupData* rtData = reinterpret_cast<SphereHitGroupData*>(optixGetSbtDataPointer());
	const int    prim_idx = optixGetPrimitiveIndex();

	float3 origin = optixGetWorldRayOrigin();
	float3 direction = optixGetWorldRayDirection();
	float t_min = optixGetRayTmin();
	float t_max = optixGetRayTmax();

	float3 oc = origin - rtData->center;
	auto a = dot(direction, direction);
	auto half_b = dot(oc, direction);
	auto c = dot(oc, oc) - rtData->radius * rtData->radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return;
	auto sqrtd = sqrt(discriminant);

	auto root = (-half_b - sqrtd) / a;
	if (root < t_min || t_max < root)
	{
		root = (-half_b + sqrtd) / a;
		if (root < t_min || t_max < root) return;
	}

	float t = root;
	float3 p = (origin + direction * root);
	float3 normal = (p - rtData->center) / rtData->radius;
	bool front_face = set_face_normal(direction, normal);

	optixReportIntersection(
		t,
		front_face ? OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE : OPTIX_HIT_KIND_TRIANGLE_BACK_FACE,
		__float_as_uint(p.x), 
		__float_as_uint(p.y), 
		__float_as_uint(p.z),
		__float_as_uint(normal.x),
		__float_as_uint(normal.y),
		__float_as_uint(normal.z)
	);
}

extern "C" __global__ void __closesthit__ch()
{
	PRD* prd = getPRD();
	if (--prd->depth <= 0)
	{
		prd->color = make_float3(0);
		return;
	}

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

	float3 target = p + random_in_hemisphere(prd->seed, normal);

	//prd->color = normal * 0.5 + 0.5;
	prd->origin = p;
	prd->direction = target - p;
	prd->hitted = true;
}

extern "C" __global__ void __miss__ray_color()
{
    float3 unit_direction = normalize(optixGetWorldRayDirection());
    auto t = 0.5 * (unit_direction.y + 1.0);

	PRD* prd = getPRD();
    prd->color = lerp(make_float3(1.0), make_float3(0.5, 0.7, 1.0), t);
	prd->hitted = false;
	prd->depth--;
}