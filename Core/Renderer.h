#pragma once

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include "Exception.h"

#include <fstream>
#include <vector>

#include <direct.h>

#include <gl/gl3w.h>

#include "Hello.h"

template <typename T>
struct SbtRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<int>        MissSbtRecord;
typedef SbtRecord<SphereHitGroupData> HitGroupSbtRecord;

#include "camera.h"


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
	std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
		<< message << "\n";
}


OptixDeviceContext context = nullptr;
OptixTraversableHandle gas_handle;
OptixModule module = nullptr;
OptixPipelineCompileOptions pipeline_compile_options = {};
OptixProgramGroup raygen_prog_group = nullptr;
OptixProgramGroup miss_prog_group = nullptr;
OptixProgramGroup hitgroup_prog_group = nullptr;
OptixPipeline pipeline = nullptr;
OptixShaderBindingTable sbt = {};

const int32_t OBJ_COUNT = 2;
static uint32_t g_obj_indices[OBJ_COUNT] = { 0, 1 };

float4* device_pixels = nullptr;
std::vector<float4> host_pixels;

// Image
const auto aspect_ratio = 16.0 / 9.0;
const int image_width = 400;
const int image_height = static_cast<int>(image_width / aspect_ratio);
const int samples_per_pixel = 100;
void Init()
{
	camera cam;

	char log[2048]; // For error reporting from OptiX creation functions

	//
	// Initialize CUDA and create OptiX context
	//
	{
		// Initialize CUDA
		CUDA_CHECK(cudaFree(0));

		CUcontext cuCtx = 0;  // zero means take the current context
		OPTIX_CHECK(optixInit());
		OptixDeviceContextOptions options = {};
		options.logCallbackFunction = &context_log_cb;
		options.logCallbackLevel = 4;
		OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
	}

	//
	// accel handling
	//
	CUdeviceptr            d_gas_output_buffer;
	{
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		// AABB build input
		OptixAabb   aabbs[OBJ_COUNT] = {
			{-1.5f, -1.5f, -1.5f, 1.5f, 1.5f, 1.5f},
			{ -100.5f, -100.5f, -100.5f, 100.5f, 100.5f, 100.5f } 
		};
		CUdeviceptr d_aabb_buffer;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), OBJ_COUNT * sizeof(OptixAabb)));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_aabb_buffer),
			aabbs,
			OBJ_COUNT * sizeof(OptixAabb),
			cudaMemcpyHostToDevice
		));

		CUdeviceptr  d_obj_indices = 0;
		const size_t obj_indices_size_in_bytes = OBJ_COUNT * sizeof(uint32_t);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_obj_indices), obj_indices_size_in_bytes));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(d_obj_indices),
			g_obj_indices,
			obj_indices_size_in_bytes,
			cudaMemcpyHostToDevice
		));

		OptixBuildInput aabb_input = {};

		aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		aabb_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
		aabb_input.customPrimitiveArray.numPrimitives = 2;

		uint32_t aabb_input_flags[2] = { OPTIX_GEOMETRY_FLAG_NONE };
		aabb_input.customPrimitiveArray.flags = aabb_input_flags;
		aabb_input.customPrimitiveArray.numSbtRecords = OBJ_COUNT;
		aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_obj_indices;
		aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &aabb_input, 1, &gas_buffer_sizes));
		CUdeviceptr d_temp_buffer_gas;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

		// non-compacted output
		CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
		size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
		CUDA_CHECK(cudaMalloc(
			reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
			compactedSizeOffset + 8
		));

		OptixAccelEmitDesc emitProperty = {};
		emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

		OPTIX_CHECK(optixAccelBuild(context,
			0,                  // CUDA stream
			&accel_options,
			&aabb_input,
			1,                  // num build inputs
			d_temp_buffer_gas,
			gas_buffer_sizes.tempSizeInBytes,
			d_buffer_temp_output_gas_and_compacted_size,
			gas_buffer_sizes.outputSizeInBytes,
			&gas_handle,
			&emitProperty,      // emitted property list
			1                   // num emitted properties
		));

		CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
		CUDA_CHECK(cudaFree((void*)d_aabb_buffer));
		CUDA_CHECK(cudaFree((void*)d_obj_indices));

		size_t compacted_gas_size;
		CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

		if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
		{
			CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), compacted_gas_size));

			// use handle as input and output
			OPTIX_CHECK(optixAccelCompact(context, 0, gas_handle, d_gas_output_buffer, compacted_gas_size, &gas_handle));

			CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
		}
		else
		{
			d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
		}
	}


	//
	// Create module
	//
	{
		OptixModuleCompileOptions module_compile_options = {};
		module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
		module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
		module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

		pipeline_compile_options.usesMotionBlur = false;
		pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pipeline_compile_options.numPayloadValues = 3;
		pipeline_compile_options.numAttributeValues = 6;
		pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
		pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

		std::string currentPath(_getcwd(NULL, 0));
		std::string filename = currentPath + "/../Binaries/x64/Debug/Hello.ptx";
		//std::string filename = currentPath + "./Hello.ptx";

		size_t inputSize = 0;
		std::fstream file(filename);
		std::string source(std::istreambuf_iterator<char>(file), {});
		const char* input = source.c_str();
		inputSize = source.size();

		size_t sizeof_log = sizeof(log);

		OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
			context,
			&module_compile_options,
			&pipeline_compile_options,
			input,
			inputSize,
			log,
			&sizeof_log,
			&module
		));
	}

	//
	// Create program groups, including NULL miss and hitgroups
	//
	{
		OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

		OptixProgramGroupDesc raygen_prog_group_desc = {}; //
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&raygen_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&raygen_prog_group
		));

		OptixProgramGroupDesc miss_prog_group_desc = {};
		miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module = module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__ray_color";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&miss_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&miss_prog_group
		));

		OptixProgramGroupDesc hitgroup_prog_group_desc = {};
		hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_desc.hitgroup.moduleCH = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
		hitgroup_prog_group_desc.hitgroup.moduleAH = nullptr;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = nullptr;
		hitgroup_prog_group_desc.hitgroup.moduleIS = module;
		hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__hit_sphere";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			context,
			&hitgroup_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&hitgroup_prog_group
		));
	}

	//
	// Link pipeline
	//
	{
		const uint32_t    max_trace_depth = 1;
		OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

		OptixPipelineLinkOptions pipeline_link_options = {};
		pipeline_link_options.maxTraceDepth = max_trace_depth;
		pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
		size_t sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixPipelineCreate(
			context,
			&pipeline_compile_options,
			&pipeline_link_options,
			program_groups,
			sizeof(program_groups) / sizeof(program_groups[0]),
			log,
			&sizeof_log,
			&pipeline
		));

		OptixStackSizes stack_sizes = {};
		for (auto& prog_group : program_groups)
		{
			OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
		}

		uint32_t direct_callable_stack_size_from_traversal;
		uint32_t direct_callable_stack_size_from_state;
		uint32_t continuation_stack_size;
		OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
			0,  // maxCCDepth
			0,  // maxDCDEpth
			&direct_callable_stack_size_from_traversal,
			&direct_callable_stack_size_from_state, &continuation_stack_size));
		OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
			direct_callable_stack_size_from_state, continuation_stack_size,
			1  // maxTraversableDepth
		));
	}

	//
	// Set up shader binding table
	//
	{
		CUdeviceptr  raygen_record;
		const size_t raygen_record_size = sizeof(RayGenSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
		RayGenSbtRecord rg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
		cam.set_sbt(rg_sbt);	
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(raygen_record),
			&rg_sbt,
			raygen_record_size,
			cudaMemcpyHostToDevice
		));

		CUdeviceptr miss_record;
		size_t      miss_record_size = sizeof(MissSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
		RayGenSbtRecord ms_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(miss_record),
			&ms_sbt,
			miss_record_size,
			cudaMemcpyHostToDevice
		));

		CUdeviceptr hitgroup_records;
		size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_records), hitgroup_record_size * OBJ_COUNT));
		HitGroupSbtRecord hg_sbts[OBJ_COUNT];
		hg_sbts[0].data.center = { 0.0f, 0.0f, -1.0f };
		hg_sbts[0].data.radius = 0.5f;
		hg_sbts[1].data.center = { 0.0f, -100.5f, -1.0f };
		hg_sbts[1].data.radius = 100;
		for (int i = 0; i < OBJ_COUNT; ++i)
		{
			OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbts[i]));
		}

		CUDA_CHECK(cudaMemcpy(
			reinterpret_cast<void*>(hitgroup_records),
			hg_sbts,
			hitgroup_record_size * OBJ_COUNT,
			cudaMemcpyHostToDevice
		));

		sbt.raygenRecord = raygen_record;
		sbt.missRecordBase = miss_record;
		sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
		sbt.missRecordCount = 1;
		sbt.hitgroupRecordBase = hitgroup_records;
		sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
		sbt.hitgroupRecordCount = OBJ_COUNT;
	}




}
 
float4* Launch(int width, int height)
{
	//
	// Create cuda device resource.
	//
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(device_pixels)));
	CUDA_CHECK(cudaMalloc(
		reinterpret_cast<void**>(&device_pixels),
		width * height * sizeof(float4)
	));

	CUstream stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	Params params;
	params.image = device_pixels;
	params.image_width = width;
	params.image_height = height;
	params.samples_per_pixel = samples_per_pixel;
	params.handle = gas_handle;

	CUdeviceptr d_param;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_param),
		&params, sizeof(params),
		cudaMemcpyHostToDevice
	));

	OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, width, height, /*depth=*/1));
	CUDA_SYNC_CHECK();

	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaStreamSynchronize(0u));


	host_pixels.resize(width * height);
	CUDA_CHECK(cudaMemcpy(
		static_cast<void*>(host_pixels.data()),
		device_pixels,
		width * height * sizeof(float4),
		cudaMemcpyDeviceToHost
	));

	return host_pixels.data();
}

void Cleanup()
{
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase)));

	OPTIX_CHECK(optixPipelineDestroy(pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
	OPTIX_CHECK(optixModuleDestroy(module));

	OPTIX_CHECK(optixDeviceContextDestroy(context));
}


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