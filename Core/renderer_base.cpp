#include <cstring>
#include <string>
#include <format>

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <spdlog/spdlog.h>

#include "exception.h"
#include "renderer_base.h"

void ContextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	// static std::string content = "";
	
	// if(strlen(message) == 0) 
	// 	return;

	// if(content.empty()) 
	// 	content = std::format("[OptiX] [{}]\n", tag);
	// content.append(message);

	// auto cr = strchr(message, '\n');
	// if(cr == nullptr)
	// {
	// 	content.append("\n");
	// }
	// else 
	// {
	// 	switch (level) 
	// 	{
	// 		case 1:  // fatal
	// 			spdlog::critical(content);
	// 			break;
	// 		case 2:  // error
	// 			spdlog::error(content);
	// 			break;
	// 		case 3:  // warning
	// 			spdlog::warn(content);
	// 			break;
	// 		case 4:  // print / info
	// 			spdlog::info(content);
	// 			break;
	// 		default: // others
	// 			spdlog::debug(content);
	// 			break;
	// 	}
	// 	content.clear();
	// }

	std::string log_msg = std::format("[OptiX][{}] {}", tag, message);
	switch (level) 
	{
		case 1:  // fatal
			spdlog::critical(log_msg);
			break;
		case 2:  // error
			spdlog::error(log_msg);
			break;
		case 3:  // warning
			spdlog::warn(log_msg);
			break;
		case 4:  // print / info
			spdlog::info(log_msg);
			break;
		default: // others
			spdlog::debug(log_msg);
			break;
	}
}

Venusaur::RendererBase::RendererBase(std::shared_ptr<OutputBuffer> outputBuffer, uint32_t maxTraceDepth) :
	m_outputBuffer(outputBuffer), m_maxTraceDepth(maxTraceDepth)
{
	CUDA_CHECK(cudaFree(0));
	CUDA_CHECK(cudaStreamCreate(&m_stream));
	
	CUcontext cuCtx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {
		.logCallbackFunction = &ContextLogCallback,
		.logCallbackData = nullptr,
		.logCallbackLevel = 4,
#ifdef _DEBUG
		// This may incur significant performance cost and should only be done during development.
		.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,
#endif
	};
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
}

Venusaur::RendererBase::~RendererBase()
{
	OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
	OPTIX_CHECK(optixDeviceContextDestroy(m_context));
}

void Venusaur::RendererBase::Draw()
{
	size_t paramsSize = UpdateParams();

	OPTIX_CHECK(optixLaunch(
		m_pipeline,
		m_stream,
		d_params,
		paramsSize,
		&m_sbt,
		m_outputBuffer->GetWidth(),
		m_outputBuffer->GetHeight(),
		1
	));
	CUDA_SYNC_CHECK();
	
	m_outputBuffer->Unmap(m_stream);

	m_outputBuffer->Display();
}


