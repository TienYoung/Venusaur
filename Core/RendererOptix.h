#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <iomanip>

#include <cuda_runtime.h>

#include <optix.h>


#include "Exception.h"
#include "RayTracer.h"

#include "Camera.h"
#include "Scene.h"
#include "CUDAOutputBuffer.h"

#include "Interfaces.h"

namespace Venusaur
{
	class RendererOptix : public IDrawable
	{
	public:
		RendererOptix(const int32_t	width, const int32_t height);
		~RendererOptix();

		void Draw() override;
		const char* GetName() const override { return m_name; }

		GLuint PBO() { return m_outputBuffer.getPBO(); }

	private:
		const char* m_name = "Optix";

		//glm::vec3 lookfrom{ 13, 2, 3 };
		//glm::vec3 lookat{ 0, 0, 0 };
		//glm::vec3 vup{ 0, 1, 0 };
		//auto dist_to_focus = 10.0f;
		//auto aperture = 0.1f;
		//const auto aspect_ratio = 3.0f / 2.0f;
		//const int image_width = 1200;
		//const int image_height = static_cast<int>(image_width / aspect_ratio);
		//Camera camera(lookfrom, 20.0f, aspect_ratio, aperture, dist_to_focus);
		Scene m_scene;

		CUDAOutputBuffer<uchar4> m_outputBuffer;

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
			OptixDeviceContext             context = nullptr;

			OptixTraversableHandle         gas_handle = 0u;
			CUdeviceptr                    d_gas_output_buffer = 0u;

			OptixModule                    ptx_module = nullptr;
			OptixPipelineCompileOptions    pipeline_compile_options = {};
			OptixPipeline                  pipeline = nullptr;
			OptixProgramGroup              raygen_prog_group = nullptr;
			OptixProgramGroup              miss_prog_group = nullptr;
			OptixProgramGroup              lambertian_hit_group = nullptr;
			OptixProgramGroup              metal_hit_group = nullptr;
			OptixProgramGroup              dielectric_hit_group = nullptr;

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

		void BuildAccelerationStructures();

		void CreateModule();

		void CreateProgramGroups();

		void CreatePipeline();

		void CreateSBT();

	};
}