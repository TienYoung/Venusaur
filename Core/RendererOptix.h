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

namespace Venusaur
{

	class RendererOptix
	{
	public:
		RendererOptix(const Scene& scene, const std::string ptxSource);
		~RendererOptix();

		void Draw(Camera& camera, CUDAOutputBuffer<uchar4>& outputBuffer);

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

		void BuildAccelerationStructures(const Scene& scene);

		void CreateModule(const std::string& ptxSource);

		void CreateProgramGroups();

		void CreatePipeline();

		void CreateSBT(const Scene& scene);

	};
}