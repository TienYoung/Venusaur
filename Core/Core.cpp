#include <cassert>
#include <chrono>
#include <filesystem>

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#include "Renderer.h"

#define ENABLE_OPTIX_RENDERER 0

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

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
Camera camera(lookfrom, 20.0f, aspect_ratio, aperture, dist_to_focus);
Scene scene;
Renderer optix_renderer;

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
	}
}

static void WindowResizeCallback(GLFWwindow* window, int width, int height)
{
	auto outputBuffer = static_cast<CUDAOutputBuffer<uchar4>*>(glfwGetWindowUserPointer(window));
	if (outputBuffer != NULL)
	{
		outputBuffer->resize(width, height);
	}
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


GLuint createGLShader(const std::string& source, GLuint shader_type)
{
	GLuint shader = glCreateShader(shader_type);
	{
		const GLchar* source_data = reinterpret_cast<const GLchar*>(source.data());
		glShaderSource(shader, 1, &source_data, nullptr);
		glCompileShader(shader);

		GLint is_compiled = 0;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
		if (is_compiled == GL_FALSE)
		{
			GLint max_length = 0;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &max_length);

			std::string info_log(max_length, '\0');
			GLchar* info_log_data = reinterpret_cast<GLchar*>(&info_log[0]);
			glGetShaderInfoLog(shader, max_length, nullptr, info_log_data);

			glDeleteShader(shader);
			std::cerr << "Compilation of shader failed: " << info_log << std::endl;

			return 0;
		}
	}


	return shader;
}

GLuint createGLProgram(
	const std::string& vert_source,
	const std::string& frag_source
)
{
	GLuint vert_shader = createGLShader(vert_source, GL_VERTEX_SHADER);
	if (vert_shader == 0)
		return 0;

	GLuint frag_shader = createGLShader(frag_source, GL_FRAGMENT_SHADER);
	if (frag_shader == 0)
	{
		glDeleteShader(vert_shader);
		return 0;
	}

	GLuint program = glCreateProgram();
	glAttachShader(program, vert_shader);
	glAttachShader(program, frag_shader);
	glLinkProgram(program);

	GLint is_linked = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
	if (is_linked == GL_FALSE)
	{
		GLint max_length = 0;
		glGetProgramiv(program, GL_INFO_LOG_LENGTH, &max_length);

		std::string info_log(max_length, '\0');
		GLchar* info_log_data = reinterpret_cast<GLchar*>(&info_log[0]);
		glGetProgramInfoLog(program, max_length, nullptr, info_log_data);
		std::cerr << "Linking of program failed: " << info_log << std::endl;

		glDeleteProgram(program);
		glDeleteShader(vert_shader);
		glDeleteShader(frag_shader);

		return 0;
	}

	glDetachShader(program, vert_shader);
	glDetachShader(program, frag_shader);


	return program;
}


int main(int argc, char* argv[])
{
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

#if ENABLE_OPTIX_RENDERER
	std::filesystem::path ptxPath(R"(E:\GitHub\Venusaur\Binaries\x64\Debug\RayTracer.ptx)");
	std::fstream ptxFile(ptxPath);
	std::string ptxSource(std::istreambuf_iterator<char>(ptxFile), {});
	ptxFile.close();
	// Init Optix.
	optix_renderer.Init(scene, ptxSource);

	int current_device, is_display_device;
	CUDA_CHECK(cudaGetDevice(&current_device));
	CUDA_CHECK(cudaDeviceGetAttribute(&is_display_device, cudaDevAttrKernelExecTimeout, current_device));
	CUDAOutputBufferType type = is_display_device ? CUDAOutputBufferType::GL_INTEROP : CUDAOutputBufferType::CUDA_DEVICE;
	CUDAOutputBuffer<uchar4> outputBuffer(type, image_width, image_height);

	glfwSetWindowUserPointer(window, &outputBuffer);
#endif // ENABLE_OPTIX_RENDERER

	glfwSetWindowAspectRatio(window, image_width, image_height);
	glfwSetWindowSizeCallback(window, WindowResizeCallback);
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

	(glTextureStorage2D(renderTex, 1, GL_RGBA8, image_width, image_height));

	auto last_time = std::chrono::steady_clock::now();
	auto current_time = last_time;

	auto end = std::chrono::steady_clock::now();
	auto start = end;

	camera.SetForward(lookat - lookfrom);

	std::string bufferName;
	
	cgltf_options options = {};
	cgltf_data* data = NULL;
	cgltf_result result = cgltf_parse_file(&options, "./CornellBox/scene.gltf", &data);
	if (result == cgltf_result_success)
	{
		cgltf_load_buffers(&options, data, "./CornellBox/scene.gltf");

		bufferName = data->buffers[0].uri;
		auto prim = data->meshes[0].primitives[0];

		auto idView = prim.indices->buffer_view;
		auto idCount = prim.indices->count;
		
		GLuint scene_ebo;
		if (prim.indices->type == cgltf_type_scalar 
			&& prim.indices->component_type == cgltf_component_type_r_32u )
		{
			if (idView->type == cgltf_buffer_view_type_indices)
			{
				glCreateBuffers(1, &scene_ebo);
				glNamedBufferStorage(scene_ebo, sizeof(unsigned int) * idCount, idView->buffer->data, GL_DYNAMIC_STORAGE_BIT);

				//auto idBuffer = idView->buffer;
				//auto idData = reinterpret_cast<unsigned int*>(idBuffer->data);
				//for (size_t i = 0; i < idCount; i++)
				//{
				//	std::cout << idData[i] << ", ";
				//}
				//std::cout << std::endl;
			}
		}

		auto vertexView = prim.attributes[0].data->buffer_view;

		auto posView = prim.attributes[1].data->buffer_view;
		auto posCount = prim.attributes[1].data->count;
		auto posOffset = prim.attributes[1].data->offset;
		auto posStride = prim.attributes[1].data->stride;

		auto normView = prim.attributes[0].data->buffer_view;
		auto normCount = prim.attributes[0].data->count;
		auto normOffset = prim.attributes[0].data->offset;
		auto normStride = prim.attributes[0].data->stride;

		auto vtData = reinterpret_cast<float*>(posView->buffer->data) + 384;
		for (size_t i = 0; i < 24; i++)
		{
			std::cout << vtData[i] << ", ";
			if ((i+1) % 3 == 0)
			{
				std::cout << std::endl;
			}
		}
		std::cout << std::endl;
		

		//GLuint scene_vbo;
		//glCreateBuffers(1, &scene_vbo);
		//glNamedBufferStorage(scene_vbo, posStride * posCount + normStride * normCount, posView->buffer->data, GL_DYNAMIC_STORAGE_BIT);

		//// VAO
		//GLuint scene_vao;
		//glCreateVertexArrays(1, &scene_vao);

		//glVertexArrayVertexBuffer(scene_vao, 0, scene_vbo, posOffset, posStride);
		//glVertexArrayVertexBuffer(scene_vao, 1, scene_vbo, normOffset, normStride);
		//glVertexArrayElementBuffer(scene_vao, scene_ebo);

		//glEnableVertexArrayAttrib(scene_vao, 0); //pos
		//glEnableVertexArrayAttrib(scene_vao, 1); //norm
		//glVertexArrayAttribFormat(scene_vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
		//glVertexArrayAttribBinding(scene_vao, 0, 0);


		cgltf_free(data);
	}

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

		float length = camera.GetFocalLength();
		ImGui::SliderFloat("Focal Length", &length, 0.0f, 20.0f);
		camera.SetFocalLength(length);
	
		//std::string scene = std::to_string(loader.GetGLTF()->at("scene").as_int64());
		//ImGui::Text("Scene:\t%s", scene.c_str());
		ImGui::Text("Buffer:\t%s", bufferName.c_str());



		ImGui::End();

		ImGui::Render();

		int width, height;
		glfwGetFramebufferSize(window, &width, &height);

		(glBindFramebuffer(GL_FRAMEBUFFER, 0));
		(glViewport(0, 0, width, height));

		const GLfloat clearColor[] = { 0.3f, 0.2f, 0.6f, 1.0f };
		const GLfloat* clearDepth = 0;
		(glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor));
		//(glClearNamedFramebufferfv(0, GL_DEPTH, 0, clearDepth));

		// Draw Optix ouput buffer.
#if ENABLE_OPTIX_RENDERER
		start = std::chrono::steady_clock::now();
		optix_renderer.Draw(camera, outputBuffer);
		end = std::chrono::steady_clock::now();
		GLuint pbo = outputBuffer.getPBO();

		(glUseProgram(program));

		// Equivalent of glActiveTexture + glBindTexture.
		glBindTextureUnit(0, renderTex);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		(glTextureSubImage2D(renderTex, 0, 0, 0, outputBuffer.width(), outputBuffer.height(), GL_RGBA, GL_UNSIGNED_BYTE, nullptr));

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glUniform1i(texLoc, 0);

		//glEnable(GL_FRAMEBUFFER_SRGB);

		glBindVertexArray(vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
#endif

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData()); 

		glfwSwapBuffers(window);

		last_time = current_time;
	}

#if ENABLE_OPTIX_RENDERER
	optix_renderer.Cleanup();
#endif

	// Cleanup.
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}