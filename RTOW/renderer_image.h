#pragma once

#include <array>

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <exception.h>

#include <renderer_base.h>

#include "image.h"

namespace RayTracingInOneWeekend
{
	class RendererImage : public Venusaur::RendererBase
	{
	public:
		RendererImage(std::shared_ptr<Venusaur::OutputBuffer> outputBuffer, const std::vector<char>& optixIR) :
			Venusaur::RendererBase(outputBuffer, 0)
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

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(ImageParams)));
		}

		~RendererImage() override
		{
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
		}

	private:
		typedef SbtRecord<void>	RayGenSbtRecord;
		typedef SbtRecord<void>	MissSbtRecord;

		size_t UpdateParams() override
		{
			ImageParams params = {};
			params.image = m_outputBuffer->Map(m_stream);
			size_t paramsSize = sizeof(ImageParams);
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, paramsSize, cudaMemcpyHostToDevice));

			return paramsSize;
		}
	};
}