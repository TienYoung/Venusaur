#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#define ENABLE_OPTIX
#undef ENABLE_OPTIX

#include <glm/glm.hpp>

#ifdef ENABLE_OPTIX
#include "Renderer.h"
#include "Scene.h"
#include "Camera.h"
#endif

#include "RendererGL.h"
#include <GLFW/glfw3.h>

static void ErrorCallback(int error, const char* description)
{
	std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}


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
Renderer optix_renderer;
#endif
std::shared_ptr<Venusaur::RendererGL> rendererGL;


static void KeyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		// Exit
		if (key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, true);
		}
	}
	if (action == GLFW_REPEAT || action == GLFW_PRESS)
	{
#ifdef ENABLE_OPTIX
		// Move
		switch (key)
		{
		case GLFW_KEY_W:
			camera.MoveForward(0.1f);
			break;
		case GLFW_KEY_S:
			camera.MoveForward(-0.1f);
			break;
		case GLFW_KEY_D:
			camera.MoveRight(0.1f);
			break;
		case GLFW_KEY_A:
			camera.MoveRight(-0.1f);
			break;
		case GLFW_KEY_Q:
			camera.MoveUp(-0.1f);
			break;
		case GLFW_KEY_E:
			camera.MoveUp(0.1f);
			break;
		case GLFW_KEY_UP:
			camera.Pitch(-1.0f);
			break;
		case GLFW_KEY_DOWN:
			camera.Pitch(1.0f);
			break;
		case GLFW_KEY_RIGHT:
			camera.Yaw(1.0f);
			break;
		case GLFW_KEY_LEFT:
			camera.Yaw(-1.0f);
			break;
		default:
			break;
		}
#endif
	}
}

static void WindowResizeCallback(GLFWwindow* window, int width, int height)
{
#ifdef ENABLE_OPTIX
	auto outputBuffer = static_cast<CUDAOutputBuffer<uchar4>*>(glfwGetWindowUserPointer(window));
	outputBuffer->resize(width, height);
#endif
}

int main(int argc, char* argv[])
{
#ifdef ENABLE_OPTIX
	std::filesystem::path ptxPath("RayTracer.ptx");
	std::fstream ptxFile(ptxPath);
	std::string ptxSource(std::istreambuf_iterator<char>(ptxFile), {});
#endif

	// Init GLFW.
	GLFWwindow* window = nullptr;
	glfwSetErrorCallback(ErrorCallback);
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_DEBUG, true);

	window = glfwCreateWindow(image_width, image_height, "Hello Optix", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwSetWindowAspectRatio(window, image_width, image_height);
	glfwSetWindowSizeCallback(window, WindowResizeCallback);
	glfwSetKeyCallback(window, KeyCallback);
	glfwMakeContextCurrent(window);
	
	// Init gl3w.
	try
	{
		rendererGL = std::make_shared<Venusaur::RendererGL>();
	}
	catch(std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

#ifdef ENABLE_OPTIX
	// Init Optix.
	optix_renderer.Init(scene, ptxSource);

	int current_device, is_display_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device));
	CUDAOutputBufferType type = is_display_device ? CUDAOutputBufferType::GL_INTEROP : CUDAOutputBufferType::CUDA_DEVICE;
	CUDAOutputBuffer<uchar4> outputBuffer(type, image_width, image_height);

	glfwSetWindowUserPointer(window, &outputBuffer);
#endif

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460 core");

	ImFont* font = io.Fonts->AddFontFromFileTTF(R"(c:\Windows\Fonts\SegoeUI.ttf)", 18.0f, NULL, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
	IM_ASSERT(font != NULL);

	auto last_time = std::chrono::steady_clock::now();
	auto current_time = last_time;

	auto end = std::chrono::steady_clock::now();
	auto start = end;
#ifdef ENABLE_OPTIX
	camera.SetForward(lookat - lookfrom);
#endif

	// Rendering.
	while (!glfwWindowShouldClose(window))
	{
		current_time = std::chrono::steady_clock::now();

		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Debugging", nullptr);
		std::chrono::duration<double> seconds = current_time - last_time;
		std::chrono::duration<double> optix_seconds = end - start;
		std::chrono::duration<double> openGL_seconds = current_time - end;
		double frame_time = seconds.count();
		double fps = 1.0 / frame_time;
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.6f, 0.1f, 1.0f));
		ImGui::Text("Frame Time:");
		ImGui::Indent(20.0f);
		ImGui::Text("OpenGL:\t%.2fms", openGL_seconds.count() * 1000);
		ImGui::Text("Optix:\t%.2fms", optix_seconds.count() * 1000);
		ImGui::Unindent(20.0f);
		ImGui::Text("FPS:%.1f\t%2fms", fps, frame_time * 1000);
		ImGui::PopStyleColor();

		float length = 10.0f;
#ifdef ENABLE_OPTIX
		length = camera.GetFocalLength();
#endif
		ImGui::SliderFloat("Focal Length", &length, 0.0f, 20.0f);
#ifdef ENABLE_OPTIX
		camera.SetFocalLength(length);
#endif
		ImGui::End();

		ImGui::Render();

		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		start = std::chrono::steady_clock::now();
#ifdef ENABLE_OPTIX
		optix_renderer.Draw(camera, outputBuffer);
#endif
		end = std::chrono::steady_clock::now();
#ifdef ENABLE_OPTIX
		GLuint pbo = outputBuffer.getPBO();
#endif

		rendererGL->Display(width, height, width, height, -1);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); 

		glfwSwapBuffers(window);

		last_time = current_time;
	}
#ifdef ENABLE_OPTIX
	optix_renderer.Cleanup();
#endif

	// Cleanup.
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}