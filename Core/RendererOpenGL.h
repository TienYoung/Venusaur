#pragma once

#include <string>

#include <GL/gl3w.h>

#include "Interfaces.h"

namespace Venusaur
{
	class RendererOpenGL : public IDrawable
	{
	public:
		RendererOpenGL(uint32_t width, uint32_t height);

		void Draw() override;

		const char* GetName() const override { return m_name; }
		GLuint GetPBO() { return m_pbo; }

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

		static const std::string s_vertexSource;
		static const std::string s_fragmentSource;
	};

} // end namespace Venusaur
