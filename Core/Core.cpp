#include "Renderer.h"

#include <cassert>
#include <chrono>

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

static void ErrorCallback(int error, const char* description)
{
	std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

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
		// Move
		switch (key)
		{
		case GLFW_KEY_W:
			cam->move_forward(0.1f);
			break;
		case GLFW_KEY_S:
			cam->move_forward(-0.1f);
			break;
		case GLFW_KEY_D:
			cam->move_right(0.1f);
			break;
		case GLFW_KEY_A:
			cam->move_right(-0.1f);
			break;
		case GLFW_KEY_Q:
			cam->move_up(-0.1f);
			break;
		case GLFW_KEY_E:
			cam->move_up(0.1f);
			break;
		case GLFW_KEY_UP:
			cam->pitch(-1.0f);
			break;
		case GLFW_KEY_DOWN:
			cam->pitch(1.0f);
			break;
		case GLFW_KEY_RIGHT:
			cam->yaw(1.0f);
			break;
		case GLFW_KEY_LEFT:
			cam->yaw(-1.0f);
			break;
		default:
			break;
		}
	}
}

static void ResizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

void message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const* message, void const* user_param)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
		case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
		case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
		case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
		case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
		case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
		case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
		case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
		case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
		case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
		case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
		case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
		case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
		case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
		case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
		case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
		case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
		case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}


const std::string s_vert_source = R"(
#version 460 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string s_frag_source = R"(
#version 460 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";

int main(int argc, char* argv[])
{


	// Init Optix.
	Init();


	// Init GLFW.
	GLFWwindow* window = nullptr;
	glfwSetErrorCallback(ErrorCallback);
	if (!glfwInit())
		return -1;

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, true);

	window = glfwCreateWindow(image_width, image_height, "Hello Optix", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwSetWindowAspectRatio(window, image_width, image_height);
	glfwSetFramebufferSizeCallback(window, ResizeCallback);
	glfwSetKeyCallback(window, KeyCallback);

	glfwMakeContextCurrent(window);
	
	// Init gl3w.
	if (gl3wInit())
	{
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	ImGui::StyleColorsDark();

	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 460 core");

	ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\SegoeUI.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
	IM_ASSERT(font != NULL);

	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(message_callback, nullptr);


	// Create resource.
	static const GLfloat vertices[] = {
		-1.0f, -1.0f , 0.0f,
		-1.0f,  1.0f , 0.0f,
		 1.0f,  1.0f , 0.0f,
		 1.0f, -1.0f , 0.0f
	};

	static const GLuint indices[] = { 0,2,1,0,3,2 };

	// VBO
	GLuint vbo;
	(glCreateBuffers(1, &vbo));
	(glNamedBufferStorage(vbo, sizeof(vertices), vertices, GL_DYNAMIC_STORAGE_BIT));

	// EBO
	GLuint ebo;
	(glCreateBuffers(1, &ebo));
	(glNamedBufferStorage(ebo, sizeof(indices), indices, GL_DYNAMIC_STORAGE_BIT));

	// VAO
	GLuint vao;
	(glCreateVertexArrays(1, &vao));

	(glEnableVertexArrayAttrib(vao, 0));
	(glVertexArrayAttribBinding(vao, 0, 0));
	(glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, 0));

	(glVertexArrayVertexBuffer(vao, 0, vbo, 0, 3 * sizeof(GLfloat)));
	glVertexArrayElementBuffer(vao, ebo);

	// PBO
	GLuint pbo;
	(glCreateBuffers(1, &pbo));

	GLuint program = createGLProgram(s_vert_source, s_frag_source);

	// Texture
	GLint texLoc = glGetUniformLocation(program, "render_tex");
	assert(texLoc != -1);
	GLuint renderTex;
	(glCreateTextures(GL_TEXTURE_2D, 1, &renderTex));

	(glTextureParameteri(renderTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	(glTextureParameteri(renderTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	(glTextureParameteri(renderTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	(glTextureParameteri(renderTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

	(glTextureStorage2D(renderTex, 1, GL_RGBA8_SNORM, image_width, image_height));


	auto last_time = std::chrono::steady_clock::now();
	auto current_time = last_time;

	// Rendering.
	while (!glfwWindowShouldClose(window))
	{
		current_time = std::chrono::steady_clock::now();

		glfwPollEvents();

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Text", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoInputs);
		std::chrono::duration<double> seconds = current_time - last_time;
		double frame_time = seconds.count();
		double fps = 1.0 / frame_time;
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.6f, 0.1f, 1.0f));
		ImGui::Text("Frame Time:%.2fms", frame_time * 1000);
		ImGui::Text("FPS:%.1f", fps);
		ImGui::PopStyleColor();
		ImGui::End();

		ImGui::Render();
		(glBindFramebuffer(GL_FRAMEBUFFER, 0));
		//(glViewport(0, 0, image_width, image_height));

		const GLfloat clearColor[] = { 0.3f, 0.2f, 0.6f, 1.0f };
		const GLfloat* clearDepth = 0;
		(glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor));
		//(glClearNamedFramebufferfv(0, GL_DEPTH, 0, clearDepth));

		(glUseProgram(program));

		UpdateHitGroupData();
		auto dataPtr = Launch(image_width, image_height, 0);
		std::vector<float4> data(dataPtr, dataPtr + image_width * image_height);
		(glNamedBufferData(pbo, sizeof(float4) * image_width * image_height, (void*)data.data(), GL_STATIC_DRAW));

		glBindTextureUnit(0, renderTex);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		(glTextureSubImage2D(renderTex, 0, 0, 0, image_width, image_height, GL_RGBA, GL_FLOAT, nullptr));

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glUniform1i(texLoc, 0);

		glEnable(GL_FRAMEBUFFER_SRGB);

		glBindVertexArray(vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); 

		glfwSwapBuffers(window);

		last_time = current_time;
	}


	// Cleanup.
	glfwDestroyWindow(window);
	glfwTerminate();

	Cleanup();

	return 0;
}