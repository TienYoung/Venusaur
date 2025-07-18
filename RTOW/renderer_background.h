#pragma once

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <glm/glm.hpp>

#include <exception.h>

#include <renderer_base.h>

#include "background.h"


namespace RayTracingInOneWeekend
{
	using glm::vec3;
	using point3 = vec3;
	
	class RendererBackground : public Venusaur::RendererBase
    {
        public:
		RendererBackground(uint32_t width, uint32_t height, const std::vector<char>& optixIR) :
			Venusaur::RendererBase(width, height, 1)
		{
			Initialize(optixIR);
		}

		~RendererBackground() override
		{
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
		}

	private:
		typedef SbtRecord<void>	RayGenSbtRecord;
		typedef SbtRecord<void>	MissSbtRecord;

		void Initialize(const std::vector<char>& optixIR) override
		{
			OptixModuleCompileOptions moduleCompileOptions = {
				.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
#ifdef OPTIX_DEBUG_DEVICE_CODE
				.optLevel 		  = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
				.debugLevel	      = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
#endif
				.boundValues 	  = nullptr,
				.numBoundValues   = 0,
				.numPayloadTypes  = 0,
				.payloadTypes     = nullptr,
			};

			OptixPipelineCompileOptions pipelineCompileOptions = {
				.usesMotionBlur = false,
				.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
				.numPayloadValues = 0,
				.numAttributeValues = 0,
				.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
				.pipelineLaunchParamsVariableName = "params",
				.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM,
				.allowOpacityMicromaps = false,
				.allowClusteredGeometry = false,
			};

			OptixModule module = nullptr;
			std::array<OptixProgramGroup, 1> programGroups = {{}};

			char   log[2048];
			size_t log_length = sizeof(log);
			OPTIX_CHECK_LOG(optixModuleCreate(
				m_context,
				&moduleCompileOptions,
				&pipelineCompileOptions,
				optixIR.data(),
				optixIR.size(),
				log,
				&log_length,
				&module
			));

			std::array<OptixProgramGroupDesc, 1> programGroupDesc = {
				OptixProgramGroupDesc{
					.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
					.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
					.raygen = {
						.module = module,
						.entryFunctionName = "__raygen__",
					}
				},
			};
			
			OptixProgramGroupOptions programGroupOptions = {
				.payloadType = nullptr,
			};
			{
				char   log[2048];
				size_t log_length = sizeof(log);
				OPTIX_CHECK_LOG(optixProgramGroupCreate(
					m_context,
					programGroupDesc.data(),
					programGroupDesc.size(),   // num program groups
					&programGroupOptions,
					log,
					&log_length,
					programGroups.data()
				));
			}

			{
				OptixPipelineLinkOptions pipelineLinkOptions = {
					.maxTraceDepth = m_maxTraceDepth,
				};

				char   log[2048];
				size_t log_length = sizeof(log);
				OPTIX_CHECK_LOG(optixPipelineCreate(
					m_context,
					&pipelineCompileOptions,
					&pipelineLinkOptions,
					programGroups.data(),
					programGroups.size(),
					log,
					&log_length,
					&m_pipeline
				));
			}

			{
				OptixStackSizes stackSizes = {};
				for (auto& progGroup : programGroups)
				{
					OPTIX_CHECK(optixUtilAccumulateStackSizes(progGroup, &stackSizes, m_pipeline));
				}
				
				uint32_t directCallableStackSizeFromTraversal;
				uint32_t directCallableStackSizeFromState;
				uint32_t continuationStackSize;
				OPTIX_CHECK(optixUtilComputeStackSizes(
					&stackSizes,
					m_maxTraceDepth,
					0,
					0,
					&directCallableStackSizeFromTraversal,
					&directCallableStackSizeFromState,
					&continuationStackSize
				));
				
				OPTIX_CHECK(optixPipelineSetStackSize(
					m_pipeline,
					directCallableStackSizeFromTraversal,
					directCallableStackSizeFromState,
					continuationStackSize,
					1
				));
			}

			CUdeviceptr  d_raygenRecord = 0;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), sizeof(RayGenSbtRecord)));
			RayGenSbtRecord raygenRecord = {};
			OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[0], &raygenRecord));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_raygenRecord),
				&raygenRecord,
				sizeof(RayGenSbtRecord),
				cudaMemcpyHostToDevice
			));

			CUdeviceptr d_missRecordBase = 0;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecordBase), sizeof(MissSbtRecord)));
			
			OPTIX_CHECK(optixProgramGroupDestroy(programGroups[0]));
			OPTIX_CHECK(optixModuleDestroy(module));

			m_sbt.raygenRecord = d_raygenRecord;
			m_sbt.missRecordBase = d_missRecordBase;
			m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
			m_sbt.missRecordCount = 1;

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(ParamsBackground)));
		}

		size_t UpdateParams() override
		{
			int image_width = m_outputBuffer->GetWidth();
			int image_height = m_outputBuffer->GetHeight();

			// Camera
			auto focal_length = 1.0;
			auto viewport_height = 2.0;
			auto viewport_width = viewport_height * (double(image_width)/image_height);
			auto camera_center = point3(0, 0, 0);
			// Calculate the vectors across the horizontal and down the vertical viewport edges.
			auto viewport_u = vec3(viewport_width, 0, 0);
			auto viewport_v = vec3(0, -viewport_height, 0);
			// Calculate the horizontal and vertical delta vectors from pixel to pixel.
			auto pixel_delta_u = viewport_u / (float)image_width;
			auto pixel_delta_v = viewport_v / (float)image_height;
			// Calculate the location of the upper left pixel.
			auto viewport_upper_left = camera_center - vec3(0, 0, focal_length) - viewport_u/2.0f - viewport_v/2.0f;
			auto pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

			ParamsBackground params = {
				.camera_center = make_float3(camera_center.x, camera_center.y, camera_center.z),
				.pixel00_loc = make_float3(pixel00_loc.x, pixel00_loc.y, pixel00_loc.z),
				.pixel_delta_u = make_float3(pixel_delta_u.x, pixel_delta_u.y, pixel_delta_u.z),
				.pixel_delta_v = make_float3(pixel_delta_v.x, pixel_delta_v.y, pixel_delta_v.z),
			};
			size_t paramsSize = sizeof(ParamsBackground);
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&params.image), nullptr, m_outputResource));
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, paramsSize, cudaMemcpyHostToDevice));

			return paramsSize;
		}
    };
}