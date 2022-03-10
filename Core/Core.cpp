#include "Renderer.h"

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <cassert>

static void ErrorCallback(int error, const char* description)
{
	std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

static void KeyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
	if (action == GLFW_PRESS)
	{
		if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
		{
			glfwSetWindowShouldClose(window, true);
		}
	}
}


const std::string s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace,1);
	UV = (vec2( vertexPosition_modelspace.x, vertexPosition_modelspace.y )+vec2(1,1))/2.0;
}
)";

const std::string s_frag_source = R"(
#version 330 core

in vec2 UV;
out vec3 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    color = texture( render_tex, UV ).xyz;
}
)";


int         width = 1920;
int         height = 1200;

int main(int argc, char* argv[])
{
	// Init Optix.
	Init();


	// Init GLFW.
	GLFWwindow* window = nullptr;
	glfwSetErrorCallback(ErrorCallback);
	if (!glfwInit())
		return -1;

	window = glfwCreateWindow(width, height, "Hello Optix", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, KeyCallback);
	
	// Init gl3w.
	if (gl3wInit())
	{
		glfwDestroyWindow(window);
		glfwTerminate();
		return -1;
	}


	// Create resource.
	GLuint vao;
	GL_CHECK(glGenVertexArrays(1, &vao));
	GL_CHECK(glBindVertexArray(vao));

	GLuint program = createGLProgram(s_vert_source, s_frag_source);
	GLint texLoc = glGetUniformLocation(program, "render_tex");
	assert(texLoc != -1);
	GLuint renderTex;
	GL_CHECK(glGenTextures(1, &renderTex));
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, renderTex));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	static const GLfloat g_quad_vertex_buffer_data[] = {
	-1.0f, -1.0f, 0.0f,
	 1.0f, -1.0f, 0.0f,
	-1.0f,  1.0f, 0.0f,

	-1.0f,  1.0f, 0.0f,
	 1.0f, -1.0f, 0.0f,
	 1.0f,  1.0f, 0.0f,
	};

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);

	GL_CHECK_ERRORS();

	auto dataPtr = Launch(width, height);
	std::vector<uchar4> data(dataPtr, dataPtr + width * height);
	GLuint pbo = 0u;
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_ARRAY_BUFFER, pbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(uchar4) * width * height, (void*)data.data(), GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	// Rendering.
	while (!glfwWindowShouldClose(window))
	{
		GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
		glViewport(0, 0, width, height);

		GL_CHECK(glClearColor(0.3f, 0.6f, 0.2f, 1.0f));
		GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

		GL_CHECK(glUseProgram(program));

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, renderTex);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glUniform1i(texLoc, 0);

		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

		glDisable(GL_FRAMEBUFFER_SRGB);

		glDrawArrays(GL_TRIANGLES, 0, 6);
		glDisableVertexAttribArray(0);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}


	// Cleanup.
	glfwDestroyWindow(window);
	glfwTerminate();

	Cleanup();

	return 0;
}