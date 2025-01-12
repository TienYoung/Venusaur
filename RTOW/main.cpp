#include <format>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

#include <GL/gl3w.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "Display.h"
#include "rt.h"

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData>			RayGenSbtRecord;
//typedef SbtRecord<MissData>				MissSbtRecord;

static void contextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << std::format("[{:2}][{:12}]: {}\n", level, tag, message);
}

int main(int argc, char* argv[])
{
	Display display = Display{ 512, 512 };

	// Create context
	OptixDeviceContext optixContext = nullptr;
	{
		cudaFree(0); // Initialize CUDA for this device on this thread
		optixInit(); // Loads the OptiX library and initializes the function table used by the stubs below.
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &contextLogCallback;
		options.logCallbackLevel = 4;
#ifdef _DEBUG
		// This may incur significant performance cost and should only be done during development.
		options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
		CUcontext cuCtx = 0; // zero means take the current context
		optixDeviceContextCreate(cuCtx, &options, &optixContext);
	}

	// Create module
	OptixModule module = nullptr;
	OptixPipelineCompileOptions pipelineCompileOptions = {};
	{
		std::filesystem::path filename = "kernel.optixir";
		if (!std::filesystem::exists(filename))
		{
			std::cout << std::filesystem::current_path();
			return 1;
		}
		std::ifstream file(filename, std::ios::binary);
		std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});

		OptixModuleCompileOptions moduleCompileOptions = {};
		moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#ifdef _DEBUG
		moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
		moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif // _DEBUG
		moduleCompileOptions.numPayloadTypes = 0;
		moduleCompileOptions.payloadTypes = nullptr;

		pipelineCompileOptions.usesMotionBlur = false;
		pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipelineCompileOptions.numPayloadValues = 0;
		pipelineCompileOptions.numAttributeValues = 2;
		pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
		pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
		pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

		char   logString[2048];
		size_t logStringSize = sizeof(logString);
		OptixResult result = optixModuleCreate(
			optixContext,
			&moduleCompileOptions,
			&pipelineCompileOptions,
			buffer.data(),
			buffer.size(),
			logString,
			&logStringSize,
			&module);
	}

	// Create Groups
	OptixProgramGroup programGroups[2] = {};
	{
		OptixProgramGroupDesc programGroupsDesc[2] = {};

		programGroupsDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		programGroupsDesc[0].raygen.module = module;
		programGroupsDesc[0].raygen.entryFunctionName = "__raygen__rg";

		programGroupsDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		programGroupsDesc[1].miss.module = module;
		programGroupsDesc[1].miss.entryFunctionName = "__miss__ms";

		char   logString[2048];
		size_t logStringSize = sizeof(logString);
		OptixProgramGroupOptions programGroupOptions = {}; // Initialize to zeros
		optixProgramGroupCreate(
			optixContext,
			programGroupsDesc,
			2,   // num program groups
			&programGroupOptions,
			logString,
			&logStringSize,
			programGroups);
	}


	// Create Pipeline
	OptixPipeline pipeline = nullptr;
	{
		OptixPipelineLinkOptions pipelineLinkOptions = {};
		pipelineLinkOptions.maxTraceDepth = 1;
		char   logString[2048];
		size_t logStringSize = sizeof(logString);
		optixPipelineCreate(
			optixContext,
			&pipelineCompileOptions,
			&pipelineLinkOptions,
			programGroups,
			2,
			logString, 
			&logStringSize,
			&pipeline);
	}

	// Create SBT
	OptixShaderBindingTable sbt = {};
	{
		CUdeviceptr  raygen_record;
		RayGenSbtRecord rg_sbt;
		cudaMalloc(reinterpret_cast<void**>(&raygen_record), sizeof(RayGenSbtRecord));
		optixSbtRecordPackHeader(programGroups[0], &rg_sbt);
		cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice);

		CUdeviceptr miss_record;
		RayGenSbtRecord ms_sbt;
		cudaMalloc(reinterpret_cast<void**>(&miss_record), sizeof(RayGenSbtRecord));
		optixSbtRecordPackHeader(programGroups[1], &ms_sbt);
		cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice);

		sbt.raygenRecord = raygen_record;
		sbt.missRecordBase = miss_record;
		sbt.missRecordStrideInBytes = sizeof(RayGenSbtRecord);
		sbt.missRecordCount = 1;
	}

	cudaGraphicsResource* pboResouce = nullptr;
	cudaGraphicsGLRegisterBuffer(&pboResouce, display.GetPBO(), cudaGraphicsMapFlagsWriteDiscard);

	Params params = {};
	params.image_width = 512;
	CUdeviceptr deviceParams = 0;
	cudaMalloc(reinterpret_cast<void**>(&deviceParams), sizeof(Params));

	while (display.IsRunning())
	{
		CUstream stream = nullptr;
		cudaStreamCreate(&stream);
		cudaGraphicsMapResources(1, &pboResouce, stream);
		size_t size = 0;
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&params.image), &size, pboResouce);
		cudaMemcpyAsync(reinterpret_cast<void*>(deviceParams), &params, sizeof(Params), cudaMemcpyHostToDevice, stream);
		optixLaunch(pipeline, stream, deviceParams, sizeof(Params), &sbt, 512, 512, 1);
		cudaGraphicsUnmapResources(1, &pboResouce, stream);
		cudaDeviceSynchronize();

		display.Draw();
	}

	optixPipelineDestroy(pipeline);
	optixProgramGroupDestroy(programGroups[0]);
	optixProgramGroupDestroy(programGroups[1]);
	optixModuleDestroy(module);
	optixDeviceContextDestroy(optixContext);

	return 0;
}