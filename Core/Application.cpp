#include <chrono>
#include <memory>

#include <GL/gl3w.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <spdlog/spdlog.h>

#include "output_buffer.h"
#include "renderer_base.h"
#include "application.h"

static void ErrorCallback(int error, const char* description)
{
	spdlog::error("[GLFW][Error {}] {}", error, description);
}

static void KeyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_RELEASE)
	{
		switch (key)
		{
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, true);
			break;
		case GLFW_KEY_TAB:
			if(glfwGetWindowAttrib(window, GLFW_DECORATED) == GLFW_TRUE)
				glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);					
			else 
				glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_TRUE);	
			break;	
		case GLFW_KEY_F1:
			static_cast<Venusaur::Application*>(glfwGetWindowUserPointer(window))->ToggleUi();
			break;
		}
	}
}

static void WindowResizeCallback(GLFWwindow* window, int width, int height)
{
	static_cast<Venusaur::Application*>(glfwGetWindowUserPointer(window))->ResizeWindow(width, height);
}

Venusaur::Application::Application(int width, int height) :
	m_width(width), m_height(height)
{
	// Init glfw.
	glfwSetErrorCallback(ErrorCallback);
	if (!glfwInit())
	{
		throw std::runtime_error("Failed to init GLFW");
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_DEBUG, true);

	m_window = glfwCreateWindow(m_width, m_height, "Venusaur", nullptr, nullptr);
	if (!m_window)
	{
		glfwTerminate();
		throw std::runtime_error("Failed to create a GLFW window!");
	}

	glfwSetKeyCallback(m_window, KeyCallback);
	glfwSetWindowUserPointer(m_window, this);
	glfwSetWindowSizeLimits(m_window, m_width, m_height, GLFW_DONT_CARE, GLFW_DONT_CARE);
	glfwSetWindowAspectRatio(m_window, m_width, m_height);
	glfwSetWindowSizeCallback(m_window, WindowResizeCallback);
	glfwMakeContextCurrent(m_window);
	
	glfwSwapInterval(1);

	if (gl3wInit())
	{
		throw std::runtime_error("Failed to initialize GL");
	}

	m_outputBuffer = std::make_shared<OutputBuffer>(m_width, m_height);

	// Init ImGui.
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL3_Init("#version 460 core");

#ifdef _WIN32
	ImFont* font = io.Fonts->AddFontFromFileTTF(R"(c:\Windows\Fonts\SegoeUI.ttf)", 18.0f, nullptr, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
	IM_ASSERT(font != nullptr);
#endif
}

Venusaur::Application::~Application()
{
	m_outputBuffer.reset();
	m_renderer.reset();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void Venusaur::Application::Update()
{
	glfwPollEvents();

	auto startPoint = std::chrono::high_resolution_clock::now();
	
	glViewport(0, 0, m_width, m_height);
	glScissor(0, 0, m_width, m_height);
	GLfloat clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GLfloat clearDepth = 0.0f;
	glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor);
	glClearNamedFramebufferfv(0, GL_DEPTH, 0, &clearDepth);
	
	m_renderer->Draw();

	auto endPoint = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endPoint - startPoint);
	
	glfwSetWindowTitle(m_window, std::format("Venusaur - {}ms", duration.count()).c_str());

	if(m_showUi)
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Debugging", nullptr);
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.6f, 0.1f, 1.0f));

		ImGui::Text("OpenGL:");
		ImGui::Indent();
		ImGui::Text("Time:\t%lldms", duration.count());
		ImGui::Text("FPS:\t%lld", 1000 / (duration.count() + 1));
		ImGui::Unindent();
		
		ImGui::PopStyleColor();
		ImGui::End();
		ImGui::EndFrame();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	glfwSwapBuffers(m_window);
}
