#include "Display.h"

#include <iostream>
#include <format>

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

	//GL_CHECK_ERRORS();

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

	//GL_CHECK_ERRORS();

	return program;
}


GLint getGLUniformLocation(GLuint program, const std::string& name)
{
	GLint loc = glGetUniformLocation(program, name.c_str());
	if (loc == -1) {
		std::cout << std::format("Failed to get uniform loc for '{}'", name).c_str();
	}
	return loc;
}

const std::string Display::s_vertexSource = R"(
		#version 460 core

		out vec2 uv0;

		void main()
		{
			uv0 = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
			gl_Position = vec4(uv0 * 2 - 1, 0, 1);
			uv0.y = 1 - uv0.y;
		}
	)";

const std::string Display::s_fragmentSource = R"(
		#version 460 core

		in vec2 uv0;
		out vec4 color;

		uniform sampler2D renderTex;

		void main()
		{
			color = texture(renderTex, uv0);
		}
	)";

void APIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const* message, void const* user_param)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << std::format("Debug message ({}): {}", id, message);

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
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behavior"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behavior"; break;
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

static void s_glfwErrorCallback(int error, const char* description)
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
}

static void s_WindowResizeCallback(GLFWwindow* window, int width, int height)
{
	//auto outputBuffer = static_cast<CUDAOutputBuffer<uchar4>*>(glfwGetWindowUserPointer(window));
	//outputBuffer->resize(width, height);
}

Display::Display(int width, int height)
	: m_width(width), m_height(height)
{
	// Init GLFW.
	glfwSetErrorCallback(s_glfwErrorCallback);
	if (!glfwInit())
	{
		std::cout << "Failed to init GLFW";
		return;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CONTEXT_DEBUG, true);

	m_window = glfwCreateWindow(m_width, m_height, "Ray Tracing in One Weekend", nullptr, nullptr);
	if (!m_window)
	{
		glfwTerminate();
		std::cout << "Failed to create a GLFW window!";
		return;
	}

	glfwSetWindowSizeCallback(m_window, s_WindowResizeCallback);
	glfwSetKeyCallback(m_window, s_KeyCallback);
	glfwMakeContextCurrent(m_window);
	glfwSwapInterval(1);

	// Init gl3w.
	if (gl3wInit())
	{
		std::cout << "Failed to initialize GL";
		return;
	}

	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(MessageCallback, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

	GLint colorSpace = 0;
	glGetNamedFramebufferAttachmentParameteriv(0, GL_FRONT_LEFT, GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, &colorSpace);
	switch (colorSpace)
	{
	case GL_RGB:
		// Enable color space convert
		glEnable(GL_FRAMEBUFFER_SRGB);
		break;
	case GL_LINEAR:
		// Disable color space convert
		glDisable(GL_FRAMEBUFFER_SRGB);
		break;
	}

	glCreateBuffers(1, &m_pbo);
	glNamedBufferData(m_pbo, 4 * sizeof(char) * m_width * m_height, nullptr, GL_STREAM_DRAW);

	// Empty VAO
	glCreateVertexArrays(1, &m_vao);

	m_program = createGLProgram(s_vertexSource, s_fragmentSource);
	m_renderTextureUniformLocation = getGLUniformLocation(m_program, "renderTex");

	// Texture
	glCreateTextures(GL_TEXTURE_2D, 1, &m_renderTex);
	glTextureParameteri(m_renderTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTextureParameteri(m_renderTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTextureParameteri(m_renderTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTextureParameteri(m_renderTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	//glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	glTextureStorage2D(m_renderTex, 1, GL_RGBA8, m_width, m_height);

}

void Display::Draw()
{
	glViewport(0, 0, m_width, m_height);

	GLfloat clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	GLfloat clearDepth = 0.0f;
	glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor);
	glClearNamedFramebufferfv(0, GL_DEPTH, 0, &clearDepth);

	glUseProgram(m_program);

	// Bind our texture in Texture Unit 0
	glBindTextureUnit(0, m_renderTex);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	//glEnable(GL_FRAMEBUFFER_SRGB);
	glTextureSubImage2D(m_renderTex, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	//glDisable(GL_FRAMEBUFFER_SRGB);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glUniform1i(m_renderTextureUniformLocation, 0);

	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);

	//GL_CHECK_ERRORS();

	glfwSwapBuffers(m_window);
	glfwPollEvents();
}