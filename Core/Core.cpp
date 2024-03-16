#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>


#define ENABLE_OPTIX
#undef ENABLE_OPTIX

#include <glm/glm.hpp>

#include "RendererGL.h"
#include "RendererOptix.h"
#include "Application.h"


// Image
glm::vec3 lookfrom{ 13, 2, 3 };
glm::vec3 lookat{ 0, 0, 0 };
glm::vec3 vup{ 0, 1, 0 };
auto dist_to_focus = 10.0f;
auto aperture = 0.1f;
const auto aspect_ratio = 3.0f / 2.0f;
const int image_width = 1200;
const int image_height = static_cast<int>(image_width / aspect_ratio);
#ifdef ENABLE_OPTIX
Camera camera(lookfrom, 20.0f, aspect_ratio, aperture, dist_to_focus);
Scene scene;
std::shared_ptr<Venusaur::RendererOptix> rendererOptix;
#endif
std::shared_ptr<Venusaur::RendererGL> rendererGL;

std::shared_ptr<Venusaur::Application> app;

int main(int argc, char* argv[])
{
#ifdef ENABLE_OPTIX
	std::filesystem::path ptxPath("RayTracer.ptx");
	std::fstream ptxFile(ptxPath);
	std::string ptxSource(std::istreambuf_iterator<char>(ptxFile), {});
#endif

	app = std::make_shared<Venusaur::Application>();

	// Init gl3w.
	try
	{
		rendererGL = std::make_shared<Venusaur::RendererGL>(image_width, image_height);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	app->AddRenderer(rendererGL);

#ifdef ENABLE_OPTIX
	// Init Optix.
	try
	{
		rendererOptix = std::make_shared<Venusaur::RendererOptix>(scene, ptxSource);
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	int current_device, is_display_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device));
	CUDAOutputBufferType type = is_display_device ? CUDAOutputBufferType::GL_INTEROP : CUDAOutputBufferType::CUDA_DEVICE;
	CUDAOutputBuffer<uchar4> outputBuffer(type, image_width, image_height);

#endif



#ifdef ENABLE_OPTIX
	camera.SetForward(lookat - lookfrom);
#endif

	// Rendering.
	while (app->IsRunning())
	{
		app->Update();
	}

	return 0;
}