#include <optix.h>

#include "Hello.h"

extern "C" 
{
    __constant__ Params params;
}

static __forceinline__ __device__ void setPayload(float3 p)
{
	optixSetPayload_0(__float_as_uint(p.x));
	optixSetPayload_1(__float_as_uint(p.y));
	optixSetPayload_2(__float_as_uint(p.z));
}

static __forceinline__ __device__ float3 getPayload()
{
	return make_float3(
		__uint_as_float(optixGetPayload_0()),
		__uint_as_float(optixGetPayload_1()),
		__uint_as_float(optixGetPayload_2())
	);
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_index = optixGetLaunchIndex();
    RayGenData* rtData = reinterpret_cast<RayGenData*>(optixGetSbtDataPointer());

    auto u = double(launch_index.x) / (params.image_width - 1);
    auto v = double(launch_index.y) / (params.image_height - 1);

    float3 origin = rtData->origin;
    float3 direction = rtData->lower_left_corner + u * rtData->horizontal + v * rtData->vertical - rtData->origin;

	// Trace the ray against our scene hierarchy
	unsigned int p0, p1, p2;
	optixTrace(
		params.handle,
		origin,
		direction,
		0.0f,                // Min intersection distance
		1e16f,               // Max intersection distance
		0.0f,                // rayTime -- used for motion blur
		OptixVisibilityMask(255), // Specify always visible
		OPTIX_RAY_FLAG_NONE,
		0,                   // SBT offset   -- See SBT discussion
		1,                   // SBT stride   -- See SBT discussion
		0,                   // missSBTIndex -- See SBT discussion
		p0, p1, p2);

	float3 pixel_color = make_float3(
		__uint_as_float(p0),
		__uint_as_float(p1),
		__uint_as_float(p2)
	);

    params.image[launch_index.y * params.image_width + launch_index.x] = make_float4(pixel_color, 1.0f);
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
	float3 normal = make_float3(
		__uint_as_float(optixGetAttribute_3()),
		__uint_as_float(optixGetAttribute_4()),
		__uint_as_float(optixGetAttribute_5()));
	setPayload(0.5 * (normal + 1));
}

extern "C" __global__ void __miss__ray_color()
{
    float3 unit_direction = normalize(optixGetWorldRayDirection());
    auto t = 0.5 * (unit_direction.y + 1.0);
    setPayload(lerp(make_float3(1.0), make_float3(0.5, 0.7, 1.0), t));
}