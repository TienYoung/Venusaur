#pragma once

#include <iostream>
#include <chrono>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "Interfaces.h"

namespace Venusaur
{
	class Application
	{

		static void s_ErrorCallback(int error, const char* description)
		{
			std::cerr << "GLFW Error " << error << ": " << description << std::endl;
		}


		static void s_KeyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
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

		static void s_WindowResizeCallback(GLFWwindow* window, int width, int height)
		{
#ifdef ENABLE_OPTIX
			auto outputBuffer = static_cast<CUDAOutputBuffer<uchar4>*>(glfwGetWindowUserPointer(window));
			outputBuffer->resize(width, height);
#endif
		}



	public:
		Application(int width = 1280, int height = 800, const char* title = "Venusaur");
		~Application();

		void AddRenderer(std::shared_ptr<IDrawable> renderer) { m_renderers.push_back(renderer); }

		void Update();
		bool IsRunning() { return glfwWindowShouldClose(m_window) == GLFW_FALSE; }
	private:
		GLFWwindow* m_window = nullptr;
		int m_width;
		int m_height;

		std::vector<std::shared_ptr<IDrawable>> m_renderers;
	};
}