#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include <cuda_runtime.h>

#include <optix.h>

#include <glm/glm.hpp>

#include "Exception.h"
#include "RayTracer.h"

#include "Camera.h"
#include "Scene.h"

#include "Interfaces.h"

namespace Venusaur
{
	class RendererOptix : public IDrawable
	{
	public:
		RendererOptix(uint32_t width, uint32_t height);
		~RendererOptix();

		void Build();

		void Draw() override;
		const char* GetName() const override { return m_name; }

		
		void SetPBO(GLuint pbo);

	private:
		const char* m_name = "Optix";

		glm::vec3 lookfrom{ 13, 2, 3 };
		glm::vec3 lookat{ 0, 0, 0 };
		glm::vec3 vup{ 0, 1, 0 };
		float dist_to_focus = 10.0f;
		float aperture = 0.1f;
		Camera camera;
		Scene m_scene;

		uint32_t m_width;
		uint32_t m_height;

		cudaGraphicsResource* m_pbo = nullptr;

		template <typename T>
		struct SbtRecord
		{
			__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
			T data;
		};

		typedef SbtRecord<RayGenData>			RayGenSbtRecord;
		typedef SbtRecord<MissData>				MissSbtRecord;
		typedef SbtRecord<MaterialData>			HitGroupSbtRecord;

		struct State
		{
			OptixDeviceContext             context = nullptr;

			OptixTraversableHandle         gas_handle = 0u;
			CUdeviceptr                    d_gas_output_buffer = 0u;

			OptixModule                    ptx_module = nullptr;
			OptixPipelineCompileOptions    pipeline_compile_options = {};
			OptixPipeline                  pipeline = nullptr;
			OptixProgramGroup              programGroups[5] = {};

			CUstream                       stream = nullptr;
			Params                         params = {};
			Params* d_params = nullptr;

			OptixShaderBindingTable        sbt = {};
		};

		State m_state = {};
		const unsigned int m_samplesPerPixel = 16;
		const uint32_t m_maxTraceDepth = 3;

		static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
		{
			std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
				<< message << "\n";
		}

		void CreateContext();

		//void BuildAccelerationStructures();

		struct SphereState
		{
			OptixTraversableHandle handle;
			OptixBuildInput input;
		};

		void BuildAccelerationStructures(SphereState& state, const float3* center, const float* radius, size_t num);

		void CreateModule(const char* filename = "RayTracer.optixir");

		void CreateProgramGroups();

		void CreatePipeline();

		void CreateSBT();

		uchar4* MapResource();

		void UnmapResource();
	};
}