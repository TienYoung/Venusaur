#include "Application.h"

#include <chrono>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "Exception.h"

#include "RendererOpenGL.h"
#include "RendererOptix.h"

Venusaur::Application::Application(int width, int height, const char* title) 
	: m_width(width), m_height(height)
{
	// Init GLFW.
	glfwSetErrorCallback(s_ErrorCallback);
	if (!glfwInit())
		throw Exception("Failed to init GLFW");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_DEBUG, true);

	m_window = glfwCreateWindow(m_width, m_height, title, nullptr, nullptr);
	if (!m_window)
	{
		glfwTerminate();
		throw Exception("Failed to create a GLFW window!");
	}

	glfwSetWindowSizeCallback(m_window, s_WindowResizeCallback);
	glfwSetKeyCallback(m_window, s_KeyCallback);
	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(1);

	// Init gl3w.
	m_rendererOpenGL = std::make_shared<RendererOpenGL>(m_width, m_height);
	m_rendererOptix = std::make_shared<RendererOptix>(m_width, m_height);
	m_rendererOptix->Build();
	m_rendererOptix->SetPBO(m_rendererOpenGL->GetPBO());
	

	// Init ImGui.
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(m_window, true);
	ImGui_ImplOpenGL3_Init("#version 460 core");

	ImFont* font = io.Fonts->AddFontFromFileTTF(R"(c:\Windows\Fonts\SegoeUI.ttf)", 18.0f, NULL, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
	IM_ASSERT(font != NULL);
}

Venusaur::Application::~Application()
{
	m_rendererOptix.reset();
	m_rendererOpenGL.reset();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(m_window);
	glfwTerminate();
}

void Venusaur::Application::Update()
{
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;
	std::chrono::duration<float, std::milli> durationMS;
	
	glfwPollEvents();

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Debugging", nullptr);
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.6f, 0.1f, 1.0f));

	try
	{
		// Optix
		begin = std::chrono::steady_clock::now();
		m_rendererOptix->Draw();
		end = std::chrono::steady_clock::now();
		durationMS = end - begin;
		ImGui::Text("Optix:\t%.2fms", durationMS.count());
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
	}

	//m_rendererOpenGL->SetPBO(m_rendererOptix->PBO());

	// OpenGL
	begin = std::chrono::steady_clock::now();
	m_rendererOpenGL->Draw();
	end = std::chrono::steady_clock::now();
	durationMS = end - begin;
	ImGui::Text("OpenGL:\t%.2fms", durationMS.count());
	
	ImGui::PopStyleColor();
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(m_window);
}
