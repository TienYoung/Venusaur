#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include <optix.h>

#include "output_buffer.h"

namespace Venusaur
{
	class RendererBase
	{
	public:
		RendererBase(std::shared_ptr<OutputBuffer> outputBuffer, uint32_t maxTraceDepth = 0);
		
		virtual ~RendererBase();
		
		void Draw();
		
	protected:
		std::shared_ptr<OutputBuffer> m_outputBuffer = nullptr;

		CUstream m_stream = nullptr;
	
		OptixDeviceContext m_context = nullptr;

		const uint32_t m_maxTraceDepth = 0;

		OptixPipeline m_pipeline = nullptr;

		OptixShaderBindingTable m_sbt = {};

		CUdeviceptr d_params = NULL;

		template <typename T>
		struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE];
			T data;
		};

		template <>
		struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord<void>
		{
			char header[OPTIX_SBT_RECORD_HEADER_SIZE];
		};

		virtual void Initialize(const std::vector<char>& optixIR) = 0;

		virtual size_t UpdateParams() = 0;
	};
}