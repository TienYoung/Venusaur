#pragma once

#include <string>

#include <cuda.h>
#include <GL/gl3w.h>
#include <cuda_gl_interop.h>

namespace Venusaur
{
	class OutputBuffer
	{
	public:
		OutputBuffer(uint32_t width, uint32_t height);
		~OutputBuffer();

		void Display();
		uchar4* Map(CUstream stream);
		void Unmap(CUstream stream);

		inline uint32_t GetWidth() const { return m_width; }
		inline uint32_t GetHeight() const { return m_height; }
	private:
		uint32_t  m_width = 0;
		uint32_t  m_height = 0;

		GLuint   m_program = 0;
		GLuint   m_vao = 0;
		GLuint   m_tex = 0;
		GLuint	 m_pbo = 0;

		cudaGraphicsResource* m_gfxResource = nullptr;

		static const std::string s_vertexSource;
		static const std::string s_fragmentSource;
	};

}
