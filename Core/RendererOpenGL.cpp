#include "RendererOpenGL.h"

#include <iostream>
#include <format>

#include "Exception.h"

namespace Venusaur
{
	namespace
	{
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
				throw Exception(std::format("Failed to get uniform loc for '{}'", name).c_str());
			}
			return loc;
		}

	} // anonymous namespace

	const std::string RendererOpenGL::s_vertexSource = R"(
		#version 460 core

		out vec2 UV;

		void main()
		{
			UV = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
			gl_Position = vec4(UV * 2 - 1, 0, 1);
		}
	)";

	const std::string RendererOpenGL::s_fragmentSource = R"(
		#version 460 core

		in vec2 UV;
		out vec3 color;

		uniform sampler2D renderTex;
		uniform bool correct_gamma;

		void main()
		{
			color = texture( renderTex, UV ).xyz;
		}
	)";

	void APIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const* message, void const* user_param)
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

	RendererOpenGL::RendererOpenGL(uint32_t width, uint32_t height)
		: m_width(width), m_height(height)
	{
		// Init gl3w.
		if (gl3wInit())
		{
			throw Exception("Failed to initialize GL");
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
		glNamedBufferData(m_pbo, sizeof(uchar4) * m_width * m_height, nullptr, GL_STREAM_DRAW);

		// Empty VAO
		glCreateVertexArrays(1, &m_vao);

		m_program = createGLProgram(s_vertexSource, s_fragmentSource);
		m_render_tex_uniform_loc = getGLUniformLocation(m_program, "renderTex");

		// Texture
		glCreateTextures(GL_TEXTURE_2D, 1, &m_renderTex);
		glTextureParameteri(m_renderTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTextureParameteri(m_renderTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(m_renderTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(m_renderTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		//glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		glTextureStorage2D(m_renderTex, 1, GL_RGBA8, m_width, m_height);

	}

	void RendererOpenGL::Draw()
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
		glUniform1i(m_render_tex_uniform_loc, 0);


		// Draw the triangles !
		glBindVertexArray(m_vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);

		//GL_CHECK_ERRORS();
	}
} // namespace Venusaur
