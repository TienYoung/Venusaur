#pragma once

#include <array>
#include <vector>

#include <cstddef>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <glm/glm.hpp>

#include <exception.h>

#include <output_buffer.h>
#include <renderer_base.h>

#include "diffuse.h"

namespace RayTracingInOneWeekend
{
	class RendererDiffuse : public Venusaur::RendererBase
    {
        public:
		RendererDiffuse(std::shared_ptr<Venusaur::OutputBuffer> outputBuffer, const std::vector<char>& optixIR) :
			Venusaur::RendererBase(outputBuffer, 1)
		{
			{
				OptixAccelBuildOptions accelBuildOption = {
					.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
					.operation = OPTIX_BUILD_OPERATION_BUILD,
					.motionOptions = {
						.numKeys = 0,
						.flags = 0,
						.timeBegin = 0.f,
						.timeEnd = 0.f,
					}
				};

				std::array sphereCenter = {
					make_float3( 0.f, 0.f, -1.f ),
					make_float3( 0.f, -100.5f, -1.f ),
				};
				
				std::array sphereRadius = {
					0.5f,
					100.0f,
				};

				CUdeviceptr d_centerBuffer;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centerBuffer), sizeof(float3) * sphereCenter.size()));
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_centerBuffer), sphereCenter.data(), sizeof(float3) * sphereCenter.size(), cudaMemcpyHostToDevice));

				CUdeviceptr d_radiusBuffer;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radiusBuffer), sizeof(float) * sphereRadius.size()));
				CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radiusBuffer), sphereRadius.data(), sizeof(float) * sphereRadius.size(), cudaMemcpyHostToDevice));

				unsigned int sphereInputFlags[] = { OPTIX_GEOMETRY_FLAG_NONE };
				OptixBuildInput sphereInput = {
					.type = OPTIX_BUILD_INPUT_TYPE_SPHERES,
					.sphereArray = {
						.vertexBuffers = &d_centerBuffer,
						.vertexStrideInBytes = sizeof(float3),
						.numVertices = sphereCenter.size(),
						.radiusBuffers = &d_radiusBuffer,
						.radiusStrideInBytes = sizeof(float),
						.singleRadius = false,
						.flags = sphereInputFlags,
						.numSbtRecords = 1,
						.sbtIndexOffsetBuffer = NULL,
						.sbtIndexOffsetSizeInBytes = 0,
						.sbtIndexOffsetStrideInBytes = 0,
						.primitiveIndexOffset = 0,
					}
				};

				OptixAccelBufferSizes gasBufferSizes = {};
				OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accelBuildOption, &sphereInput, 1, &gasBufferSizes));
				
				CUdeviceptr d_gasTempBuffer = NULL;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasTempBuffer), gasBufferSizes.tempSizeInBytes));
				
				CUdeviceptr d_gasOutputBuffer = NULL;
				CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutputBuffer), gasBufferSizes.outputSizeInBytes));

				OPTIX_CHECK(optixAccelBuild(
					m_context,
					m_stream, 
					&accelBuildOption,
					&sphereInput,
					1,
					d_gasTempBuffer,
					gasBufferSizes.tempSizeInBytes,
					d_gasOutputBuffer,
					gasBufferSizes.outputSizeInBytes,
					&m_gasHandle,
					nullptr,
					0
				));

				CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasTempBuffer)));
				d_gasBuffer = d_gasOutputBuffer;
			}

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
				.numPayloadValues = sizeof(DiffusePayload) / sizeof(unsigned int),
				.numAttributeValues = 0,
				.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
				.pipelineLaunchParamsVariableName = "params",
				.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE,
				.allowOpacityMicromaps = false,
				.allowClusteredGeometry = false,
			};

			OptixModule module = nullptr;

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

			OptixModule sphereModuleIS = nullptr;
			OptixBuiltinISOptions sphereISOptions = {
				.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE,
				.usesMotionBlur = false,
				.buildFlags = OPTIX_BUILD_FLAG_NONE,
				.curveEndcapFlags = OPTIX_CURVE_ENDCAP_DEFAULT,
			};
			OPTIX_CHECK_LOG(optixBuiltinISModuleGet(
				m_context,
				&moduleCompileOptions,
				&pipelineCompileOptions,
				&sphereISOptions,
				&sphereModuleIS
			));

			std::array programGroupDesc = {
				OptixProgramGroupDesc{
					.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
					.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
					.raygen = {
						.module = module,
						.entryFunctionName = "__raygen__",
					}
				},
				OptixProgramGroupDesc{
					.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
					.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
					.miss = {
						.module = module,
						.entryFunctionName = "__miss__",
					}
				},
				OptixProgramGroupDesc{
					.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
					.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
					.hitgroup = {
						.moduleCH = module,
						.entryFunctionNameCH = "__closesthit__",
						.moduleAH = nullptr,
						.entryFunctionNameAH = nullptr,
						.moduleIS = sphereModuleIS,
						.entryFunctionNameIS = nullptr,
					}
				}
			};
			
			std::array<OptixProgramGroup, programGroupDesc.size()> programGroups = {};

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

			CUdeviceptr  d_raygenRecord = NULL;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), sizeof(RayGenSbtRecord)));
			RayGenSbtRecord raygenRecord = {};
			OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[0], &raygenRecord));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_raygenRecord),
				&raygenRecord,
				sizeof(RayGenSbtRecord),
				cudaMemcpyHostToDevice
			));
			OPTIX_CHECK(optixProgramGroupDestroy(programGroups[0]));
			
			CUdeviceptr d_missRecordBase = NULL;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecordBase), sizeof(MissSbtRecord)));
			MissSbtRecord missRecord = {};
			OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[1], &missRecord));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_missRecordBase),
				&missRecord,
				sizeof(MissSbtRecord),
				cudaMemcpyHostToDevice
			));
			OPTIX_CHECK(optixProgramGroupDestroy(programGroups[1]));

			CUdeviceptr d_hitGroupRecordBase = NULL;
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitGroupRecordBase), sizeof(HitGroupSbtRecord)));
			HitGroupSbtRecord hitGroupRecord = {};
			OPTIX_CHECK(optixSbtRecordPackHeader(programGroups[2], &hitGroupRecord));
			CUDA_CHECK(cudaMemcpy(
				reinterpret_cast<void*>(d_hitGroupRecordBase),
				&hitGroupRecord,
				sizeof(HitGroupSbtRecord),
				cudaMemcpyHostToDevice
			));
			OPTIX_CHECK(optixProgramGroupDestroy(programGroups[2]));
			
			OPTIX_CHECK(optixModuleDestroy(module));

			m_sbt = {
				.raygenRecord = d_raygenRecord,
				.missRecordBase = d_missRecordBase,
				.missRecordStrideInBytes = sizeof(MissSbtRecord),
				.missRecordCount = 1,
				.hitgroupRecordBase = d_hitGroupRecordBase,
				.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord),
				.hitgroupRecordCount = 1,
				.callablesRecordBase = NULL,
				.callablesRecordStrideInBytes = 0,
				.callablesRecordCount = 0,
			};

			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(DiffuseParams)));
		}

		~RendererDiffuse() override
		{
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasBuffer)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
			CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));
		}

	private:
		typedef SbtRecord<void>	RayGenSbtRecord;
		typedef SbtRecord<void>	MissSbtRecord;
		typedef SbtRecord<void> HitGroupSbtRecord;

		OptixTraversableHandle m_gasHandle = 0;
		CUdeviceptr d_gasBuffer = NULL;

		unsigned int m_subframeIndex = 0;

		size_t UpdateParams() override
		{
			int image_width = m_outputBuffer->GetWidth();
			int image_height = m_outputBuffer->GetHeight();

			// Camera
			auto focal_length = 1.0;
			auto viewport_height = 2.0;
			auto viewport_width = viewport_height * (double(image_width)/image_height);
			auto camera_center = glm::vec3(0, 0, 0);
			// Calculate the vectors across the horizontal and down the vertical viewport edges.
			auto viewport_u = glm::vec3(viewport_width, 0, 0);
			auto viewport_v = glm::vec3(0, -viewport_height, 0);
			// Calculate the horizontal and vertical delta vectors from pixel to pixel.
			auto pixel_delta_u = viewport_u / (float)image_width;
			auto pixel_delta_v = viewport_v / (float)image_height;
			// Calculate the location of the upper left pixel.
			auto viewport_upper_left = camera_center - glm::vec3(0, 0, focal_length) - viewport_u/2.0f - viewport_v/2.0f;
			auto pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

			DiffuseParams params = {
				.image = m_outputBuffer->Map(m_stream),
				.camera_center = make_float3(camera_center.x, camera_center.y, camera_center.z),
				.pixel00_loc = make_float3(pixel00_loc.x, pixel00_loc.y, pixel00_loc.z),
				.pixel_delta_u = make_float3(pixel_delta_u.x, pixel_delta_u.y, pixel_delta_u.z),
				.pixel_delta_v = make_float3(pixel_delta_v.x, pixel_delta_v.y, pixel_delta_v.z),
				.samples_per_pixel = 100,
				.subframe_index = m_subframeIndex++,
				.handle = m_gasHandle,
			};
			size_t paramsSize = sizeof(DiffuseParams);
			CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, paramsSize, cudaMemcpyHostToDevice));

			return paramsSize;
		}
    };
}