//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "RendererGL.h"

#include <iostream>
#include <format>

namespace Venusaur
{

	//-----------------------------------------------------------------------------
	//
	// Helper functions
	//
	//-----------------------------------------------------------------------------
	namespace
	{

		size_t pixelFormatSize(BufferImageFormat format)
		{
			switch (format)
			{
			case BufferImageFormat::UNSIGNED_BYTE4:
				return sizeof(char) * 4;
			case BufferImageFormat::FLOAT4:
				return sizeof(float) * 4;
			case BufferImageFormat::FLOAT3:
				return sizeof(float) * 3;
			default:
				throw Exception("sutil::pixelFormatSize: Unrecognized buffer format");
			}
		}

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


	//-----------------------------------------------------------------------------
	//
	// GLDisplay implementation
	//
	//-----------------------------------------------------------------------------

	const std::string RendererGL::s_vert_source = R"(
		#version 460 core

		layout(location = 0) in vec3 vertexPosition_modelspace;
		out vec2 UV;

		void main()
		{
			gl_Position =  vec4(vertexPosition_modelspace,1);
			UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
		}
	)";

	const std::string RendererGL::s_frag_source = R"(
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

	RendererGL::RendererGL(BufferImageFormat image_format)
		: m_image_format(image_format)
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

		static const GLfloat vertices[] = {
			-1.0f, -1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f,
			-1.0f,  1.0f, 0.0f,

			-1.0f,  1.0f, 0.0f,
			 1.0f, -1.0f, 0.0f,
			 1.0f,  1.0f, 0.0f,
		};

		// VAO
		glCreateVertexArrays(1, &m_vao);

		// VBO
		glCreateBuffers(1, &m_vbo);
		glNamedBufferStorage(m_vao, sizeof(vertices), vertices, GL_DYNAMIC_STORAGE_BIT);

		// 1st attribute buffer : vertices
		glVertexArrayVertexBuffer(m_vao, 0, m_vbo, 0, sizeof(GLfloat) * 3);
		glEnableVertexArrayAttrib(m_vao, 0);
		glVertexArrayAttribFormat(m_vao, 0, 3, GL_FLOAT, GL_FALSE, 0);
		glDisableVertexArrayAttrib(m_vao, 0);

		// Texture
		glCreateTextures(GL_TEXTURE_2D, 1, &m_renderTex);
		glTextureParameteri(m_renderTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTextureParameteri(m_renderTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(m_renderTex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(m_renderTex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		m_program = createGLProgram(s_vert_source, s_frag_source);
		m_render_tex_uniform_loc = getGLUniformLocation(m_program, "renderTex");
	}


	void RendererGL::Display(
		const int32_t  screen_res_x,
		const int32_t  screen_res_y,
		const int32_t  framebuf_res_x,
		const int32_t  framebuf_res_y,
		const uint32_t pbo
	) const
	{
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, framebuf_res_x, framebuf_res_y);

		GLfloat clearColor[] = { 0.3f, 0.2f, 0.6f, 1.0f };
		GLfloat clearDepth = 0.0f;
		glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor);
		glClearNamedFramebufferfv(0, GL_DEPTH, 0, &clearDepth);

		glUseProgram(m_program);

		// Bind our texture in Texture Unit 0
		glBindTextureUnit(0, m_renderTex);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

		size_t elmt_size = pixelFormatSize(m_image_format);
		if (elmt_size % 8 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
		else if (elmt_size % 4 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		else if (elmt_size % 2 == 0) glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
		else                          glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		bool convertToSrgb = true;

		if (m_image_format == BufferImageFormat::UNSIGNED_BYTE4)
		{
			glTextureStorage2D(m_renderTex, 1, GL_RGBA8, screen_res_x, screen_res_y);
			// input is assumed to be in sRGB since it is only 1 byte per channel in size
			glTextureSubImage2D(m_renderTex, 0, 0, 0, screen_res_x, screen_res_y, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			convertToSrgb = false;
		}
		else if (m_image_format == BufferImageFormat::FLOAT3)
		{
			glTextureStorage2D(m_renderTex, 1, GL_RGB8_SNORM, screen_res_x, screen_res_y);
			glTextureSubImage2D(m_renderTex, 0, 0, 0, screen_res_x, screen_res_y, GL_RGB, GL_FLOAT, nullptr);
		}
		else if (m_image_format == BufferImageFormat::FLOAT4) 
		{
			glTextureStorage2D(m_renderTex, 1, GL_RGBA8_SNORM, screen_res_x, screen_res_y);
			glTextureSubImage2D(m_renderTex, 0, 0, 0, screen_res_x, screen_res_y, GL_RGBA, GL_FLOAT, nullptr);
		}
		else
			throw Exception("Unknown buffer format");

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glUniform1i(m_render_tex_uniform_loc, 0);

		if (convertToSrgb)
			glEnable(GL_FRAMEBUFFER_SRGB);
		else
			glDisable(GL_FRAMEBUFFER_SRGB);

		// Draw the triangles !
		glBindVertexArray(m_vao);
		glDrawArrays(GL_TRIANGLES, 0, 6); // 2*3 indices starting at 0 -> 2 triangles

		glDisable(GL_FRAMEBUFFER_SRGB);

		//GL_CHECK_ERRORS();
	}

} // namespace sutil
