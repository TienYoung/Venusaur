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



#pragma once

#include <cstdint>
#include <string>

#include <GL/gl3w.h>

#include "Interfaces.h"
#include "Exception.h"

namespace Venusaur
{

	enum BufferImageFormat
	{
		UNSIGNED_BYTE4,
		FLOAT4,
		FLOAT3
	};

	class RendererOpenGL : public IDrawable
	{
	public:
		RendererOpenGL(const int32_t width, const int32_t height, BufferImageFormat format = BufferImageFormat::UNSIGNED_BYTE4);

		void Draw() const override;

		const char* GetName() const override { return m_name; }

	private:
		const char* m_name = "OpenGL";

		GLsizei  m_width;
		GLsizei  m_height;

		GLuint   m_renderTex = 0u;
		GLuint   m_program = 0u;
		GLint    m_render_tex_uniform_loc = -1;
		GLuint   m_vbo = 0;
		GLuint   m_vao = 0;

		GLuint	 m_pbo = 0;

		BufferImageFormat m_image_format;

		static const std::string s_vert_source;
		static const std::string s_frag_source;
	};

} // end namespace Venusaur
