#pragma once

#include <string>

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

class Display
{
public:
	Display(uint32_t width = 512, uint32_t height = 512);
	Display() = delete;

	bool IsRunning()
	{
		return !glfwWindowShouldClose(m_window);
	}

	void Draw();

	GLuint GetPBO() const { return m_pbo; }

private:
	GLFWwindow* m_window = nullptr;
	GLsizei  m_width;
	GLsizei  m_height;

	GLuint   m_renderTex = 0u;
	GLuint   m_program = 0u;
	GLint    m_renderTextureUniformLocation = -1;
	GLuint   m_vbo = 0;
	GLuint   m_vao = 0;

	GLuint	 m_pbo = 0;

	static const std::string s_vertexSource;
	static const std::string s_fragmentSource;
};