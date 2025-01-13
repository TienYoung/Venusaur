#include <optix.h>

#include <OptiXToolkit/ShaderUtil/color.h>

#include "rt.h"

extern "C" {
	__constant__ Params params;
}

extern "C" __global__ void __raygen__uv()
{
	const uint3 launch_index = optixGetLaunchIndex();
	int i = launch_index.x;
	int j = launch_index.y;

	int image_width = params.width;
	int image_height = params.height;

	auto r = double(i) / (image_width - 1);
	auto g = double(j) / (image_height - 1);
	auto b = 0.0;

	int ir = int(255.999 * r);
	int ig = int(255.999 * g);
	int ib = int(255.999 * b);

	params.image[j * image_width + i] = make_uchar4(ir, ig, ib, 255u);
}