#include <format>

#include <spdlog/spdlog.h>

#include "exception.h"

#include "output_buffer.h"

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
					GLint log_length = 0;
					glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

					std::string info_log(log_length, '\0');
					glGetShaderInfoLog(shader, log_length, nullptr, info_log.data());
					
					spdlog::error("[Shader Compile Error] {}", info_log);
					glDeleteShader(shader);

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
				GLint log_length = 0;
				glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);

				std::string info_log(log_length, '\0');
				glGetProgramInfoLog(program, log_length, nullptr, info_log.data());
				
				spdlog::error("[Program Link Error] {}", info_log);
				glDeleteProgram(program);
				glDeleteShader(vert_shader);
				glDeleteShader(frag_shader);

				return 0;
			}

			glDetachShader(program, vert_shader);
			glDetachShader(program, frag_shader);

			return program;
		}


		GLint getGLUniformLocation(GLuint program, const std::string& name)
		{
			GLint loc = glGetUniformLocation(program, name.c_str());
			if (loc == -1) {
				throw std::runtime_error(std::format("Failed to get uniform loc for '{}'", name).c_str());
			}
			return loc;
		}

	} // anonymous namespace

	const std::string OutputBuffer::s_vertexSource = R"(
		#version 460 core

		out vec2 texcoord;

		void main()
		{
			texcoord = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
			gl_Position = vec4(texcoord * 2 - 1, 0, 1);
		}
	)";

	const std::string OutputBuffer::s_fragmentSource = R"(
		#version 460 core

		layout(binding = 0) uniform sampler2D tex;
		in vec2 texcoord;
		out vec4 color;
		
		void main()
		{
			vec2 uv = vec2(texcoord.x, 1.0 - texcoord.y);
			color = texture(tex, uv);
		}
	)";

	void APIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const* message, void const* user_param)
	{
		// ignore non-significant error/warning codes
		if (id == 131154 || /*id == 131169 ||*/ id == 131185 /*|| id == 131218 || id == 131204*/)
			return;

		std::string sourceStr, typeStr, severityStr;

		switch (source)	{
			case GL_DEBUG_SOURCE_API:             sourceStr = "API"; break;
			case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   sourceStr = "Window System"; break;
			case GL_DEBUG_SOURCE_SHADER_COMPILER: sourceStr = "Shader Compiler"; break;
			case GL_DEBUG_SOURCE_THIRD_PARTY:     sourceStr = "Third Party"; break;
			case GL_DEBUG_SOURCE_APPLICATION:     sourceStr = "Application"; break;
			case GL_DEBUG_SOURCE_OTHER:           sourceStr = "Other"; break;
			default:							  sourceStr = "Unknown"; break;
		} 

		switch (type) {
			case GL_DEBUG_TYPE_ERROR:               typeStr = "Error"; break;
			case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typeStr = "Deprecated Behavior"; break;
			case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typeStr = "Undefined Behavior"; break;
			case GL_DEBUG_TYPE_PORTABILITY:         typeStr = "Portability"; break;
			case GL_DEBUG_TYPE_PERFORMANCE:         typeStr = "Performance"; break;
			case GL_DEBUG_TYPE_MARKER:              typeStr = "Marker"; break;
			case GL_DEBUG_TYPE_PUSH_GROUP:          typeStr = "Push Group"; break;
			case GL_DEBUG_TYPE_POP_GROUP:           typeStr = "Pop Group"; break;
			case GL_DEBUG_TYPE_OTHER:               typeStr = "Other"; break;
			default:								typeStr = "Unknown"; break;
		}

		switch (severity) {
			case GL_DEBUG_SEVERITY_HIGH: 
				spdlog::critical("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
				break;
			case GL_DEBUG_SEVERITY_MEDIUM:
				spdlog::error("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
				break;
			case GL_DEBUG_SEVERITY_LOW:
				spdlog::warn("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
				break;
			case GL_DEBUG_SEVERITY_NOTIFICATION:
				spdlog::info("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
				break;
			default:
				spdlog::debug("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
				break;
		}
	}

	OutputBuffer::OutputBuffer(uint32_t width, uint32_t height) : 
		m_width(width), m_height(height)
	{
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(MessageCallback, nullptr);
		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

		glDisable(GL_BLEND);
		glDisable(GL_CULL_FACE);
		glDisable(GL_DEPTH_TEST);

		GLint colorSpace = 0;
		glGetNamedFramebufferAttachmentParameteriv(0, GL_FRONT_LEFT, GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, &colorSpace);
		switch (colorSpace) 
		{
		case GL_RGB:
			glEnable(GL_FRAMEBUFFER_SRGB);
			break;
		case GL_LINEAR:
			glDisable(GL_FRAMEBUFFER_SRGB);
			break;
		}

		// Create program
		m_program = createGLProgram(s_vertexSource, s_fragmentSource);
		glUseProgram(m_program);

		// Create VAO
		glCreateVertexArrays(1, &m_vao);

		// Create Texture
		glCreateTextures(GL_TEXTURE_2D, 1, &m_tex);
		glTextureStorage2D(m_tex, 1, GL_RGBA8, m_width, m_height);
		glTextureParameteri(m_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTextureParameteri(m_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(m_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(m_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glBindTextureUnit(0, m_tex);

		// Create PBO
		glCreateBuffers(1, &m_pbo);
		glNamedBufferData(m_pbo, sizeof(uint8_t) * 4 * m_width * m_height, nullptr, GL_STREAM_DRAW);

		CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_gfxResource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));
	}
	
	OutputBuffer::~OutputBuffer()
	{
		CUDA_CHECK(cudaGraphicsUnregisterResource(m_gfxResource));	
	}

	void OutputBuffer::Display()
	{
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
		glPixelStorei(GL_UNPACK_ALIGNMENT,4);
		glTextureSubImage2D(m_tex, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glBindVertexArray(m_vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	uchar4* OutputBuffer::Map(CUstream stream)
	{
		uchar4* image = nullptr;
		CUDA_CHECK(cudaGraphicsMapResources(1, &m_gfxResource, stream));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&image), nullptr, m_gfxResource));
		return image;
	}
	
	void OutputBuffer::Unmap(CUstream stream)
	{
		CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_gfxResource, stream));
	}
}
