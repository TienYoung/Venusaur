#pragma once

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include "Exception.h"

#include <fstream>
#include <vector>
#include <string>

#include <direct.h>

#include <gl/gl3w.h>

#include "RayTracer.h"
#include "Scene.h"
#include "Camera.h"

class Renderer
{
public:
	Renderer();
	~Renderer();

private:
	template <typename T>
	struct SbtRecord
	{
		__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		T data;
	};

	typedef SbtRecord<RayGenData>         RayGenSbtRecord;
	typedef SbtRecord<MissData>           MissSbtRecord;
	typedef SbtRecord<SphereHitGroupData> HitGroupSbtRecord;

	struct State
	{
		OptixDeviceContext             context                  = nullptr;

		OptixTraversableHandle         gas_handle               = 0u;
		CUdeviceptr                    d_gas_output_buffer      = 0u;
		
		OptixModule                    ptx_module               = nullptr;
		OptixPipelineCompileOptions    pipeline_compile_options = {};
		OptixPipeline                  pipeline                 = nullptr;
		OptixProgramGroup              raygen_prog_group        = nullptr;
		OptixProgramGroup              miss_prog_group          = nullptr;
		OptixProgramGroup              lambertian_hit_group     = nullptr;
		OptixProgramGroup              metal_hit_group          = nullptr;
		OptixProgramGroup              dielectric_hit_group     = nullptr;

		CUstream                       stream                   = nullptr;
		Params                         params                   = {};
		Params*                        d_params                 = nullptr;

		OptixShaderBindingTable        sbt                      = {};
	};

	State m_state                        = {};
	const unsigned int m_samplesPerPixel = 16;
	const uint32_t m_maxTraceDepth       = 3;

	static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
	{
		std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
			<< message << "\n";
	}

	void CreateContext();
	void BuildAccelerationStructures(const Scene& scene);
	void CreateModule(const std::string& ptxSource);
	void CreateProgramGroups();
	void CreatePipeline();
	void CreateSBT(const Scene& scene);

	void Launch(const Camera& camera);
};

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