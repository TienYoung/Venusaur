#include "RendererOptix.h"

#include <filesystem>

#include <cuda_gl_interop.h>

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "vec_math.h"

Venusaur::RendererOptix::RendererOptix(uint32_t width, uint32_t height)
	: camera(lookfrom, 20.0f, (float)width / height, aperture, dist_to_focus), m_width(width), m_height(height)
{
	CreateContext();
}

void Venusaur::RendererOptix::Build()
{
	//BuildAccelerationStructures();
	SphereState sphereState = {};
	const float3* p_centers = nullptr;
	const float* p_radius = nullptr;
	size_t num = m_scene.getSpheresData(p_centers, p_radius);
	BuildAccelerationStructures(sphereState, p_centers, p_radius, num);

	CreateModule();
	CreateProgramGroups();
	CreatePipeline();
	CreateSBT();

	m_state.params.subframe_index = 0u;

	//m_state.params.image = m_outputBuffer.map();
	//m_state.params.width = m_outputBuffer.width();
	//m_state.params.height = m_outputBuffer.height();
	m_state.params.width = m_width;
	m_state.params.height = m_height;
	//m_state.params.samples_per_pixel = 16;
	//m_state.params.subframe_index++;
	glm::vec3 origin = camera.GetPosition();
	m_state.params.origin = make_float3(origin.x, origin.y, origin.z);
	glm::vec3 u, v, w;
	camera.UVWFrame(u, v, w);
	m_state.params.u = make_float3(u.x, u.y, u.z);
	m_state.params.v = make_float3(v.x, v.y, v.z);
	m_state.params.w = make_float3(w.x, w.y, w.z);
	m_state.params.lens_radius = camera.GetLensRadius();

	m_state.params.handle = m_state.gas_handle;

	CUDA_CHECK(cudaStreamCreate(&m_state.stream));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_params), sizeof(Params)));
}

Venusaur::RendererOptix::~RendererOptix()
{
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.hitgroupRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_gas_output_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.params.accum)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_params)));

	CUDA_CHECK(cudaGraphicsUnregisterResource(m_pbo));

	OPTIX_CHECK(optixPipelineDestroy(m_state.pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.programGroups[0]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.programGroups[1]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.programGroups[2]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.programGroups[3]));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.programGroups[4]));
	OPTIX_CHECK(optixModuleDestroy(m_state.ptx_module));
	OPTIX_CHECK(optixDeviceContextDestroy(m_state.context));
}

void Venusaur::RendererOptix::Draw()
{
	uchar4* result_buffer_data = MapResource();
	m_state.params.image = result_buffer_data;
	CUDA_CHECK(cudaMemcpyAsync(
		reinterpret_cast<void*>(m_state.d_params),
		&m_state.params, sizeof(Params),
		cudaMemcpyHostToDevice, m_state.stream
	));

	OPTIX_CHECK(optixLaunch(
		m_state.pipeline,
		m_state.stream,
		reinterpret_cast<CUdeviceptr>(m_state.d_params),
		sizeof(Params),
		&m_state.sbt,
		m_width,
		m_height,
		/*depth=*/1
	));
	UnmapResource();
	CUDA_SYNC_CHECK();
}

void Venusaur::RendererOptix::SetPBO(GLuint pbo)
{
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard));
}

void Venusaur::RendererOptix::CreateContext()
{
	//
	// Initialize CUDA and create OptiX context
	//
	CUDA_CHECK(cudaFree(0));
	CUcontext cuCtx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction = &context_log_cb;
	options.logCallbackLevel = 4;
#ifdef _DEBUG
	// This may incur significant performance cost and should only be done during development.
	options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#endif
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_state.context));
}

//void Venusaur::RendererOptix::BuildAccelerationStructures()
//{
//	OptixAccelBuildOptions accel_options = {};
//	accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
//	accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
//
//	// AABB build input
//	const size_t aabbs_size_in_bytes = m_scene.m_aabbs.size() * sizeof(OptixAabb);
//	CUdeviceptr d_aabb_buffer;
//	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), aabbs_size_in_bytes));
//	CUDA_CHECK(cudaMemcpy(
//		reinterpret_cast<void*>(d_aabb_buffer),
//		m_scene.m_aabbs.data(),
//		aabbs_size_in_bytes,
//		cudaMemcpyHostToDevice
//	));
//
//	const size_t obj_indices_size_in_bytes = m_scene.m_indices.size() * sizeof(uint32_t);
//	CUdeviceptr  d_obj_indices = 0;
//	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_obj_indices), obj_indices_size_in_bytes));
//	CUDA_CHECK(cudaMemcpy(
//		reinterpret_cast<void*>(d_obj_indices),
//		m_scene.m_indices.data(),
//		obj_indices_size_in_bytes,
//		cudaMemcpyHostToDevice
//	));
//
//	OptixBuildInput aabb_input = {};
//	aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
//	aabb_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
//	aabb_input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(m_scene.m_spheres.size());
//
//	unsigned int* aabb_input_flags = new unsigned int[m_scene.m_spheres.size()];
//	for (size_t i = 0; i < m_scene.m_spheres.size(); ++i)
//	{
//		aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
//	}
//	aabb_input.customPrimitiveArray.flags = aabb_input_flags;
//	aabb_input.customPrimitiveArray.numSbtRecords = static_cast<unsigned int>(m_scene.m_spheres.size());
//	aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_obj_indices;
//	aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
//
//	OptixAccelBufferSizes gas_buffer_sizes;
//	OPTIX_CHECK(optixAccelComputeMemoryUsage(m_state.context, &accel_options, &aabb_input, 1, &gas_buffer_sizes));
//	CUdeviceptr d_temp_buffer_gas;
//	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
//
//	// non-compacted output
//	CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
//	size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
//	CUDA_CHECK(cudaMalloc(
//		reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
//		compactedSizeOffset + 8
//	));
//
//	OptixAccelEmitDesc emitProperty = {};
//	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
//	emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);
//
//	OPTIX_CHECK(optixAccelBuild(
//		m_state.context,
//		0,                  // CUDA stream
//		&accel_options,
//		&aabb_input,
//		1,                  // num build inputs
//		d_temp_buffer_gas,
//		gas_buffer_sizes.tempSizeInBytes,
//		d_buffer_temp_output_gas_and_compacted_size,
//		gas_buffer_sizes.outputSizeInBytes,
//		&m_state.gas_handle,
//		&emitProperty,      // emitted property list
//		1                   // num emitted properties
//	));
//
//	CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
//	CUDA_CHECK(cudaFree((void*)d_aabb_buffer));
//	CUDA_CHECK(cudaFree((void*)d_obj_indices));
//	delete[] aabb_input_flags;
//
//	size_t compacted_gas_size;
//	CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));
//
//	if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
//	{
//		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_gas_output_buffer), compacted_gas_size));
//
//		// use handle as input and output
//		OPTIX_CHECK(optixAccelCompact(m_state.context, 0, m_state.gas_handle, m_state.d_gas_output_buffer, compacted_gas_size, &m_state.gas_handle));
//
//		CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
//	}
//	else
//	{
//		m_state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
//	}
//}

void Venusaur::RendererOptix::BuildAccelerationStructures(SphereState& state, const float3* center, const float* radius, size_t num)
{
	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;


	CUdeviceptr centerBuffer = NULL;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&centerBuffer), num * sizeof(float3)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(centerBuffer),
		center,
		num * sizeof(float3),
		cudaMemcpyHostToDevice
	));

	CUdeviceptr radiusBuffer = NULL;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&radiusBuffer), num * sizeof(float)));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(radiusBuffer),
		radius,
		num * sizeof(float),
		cudaMemcpyHostToDevice
	));

	state.input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
	state.input.sphereArray.vertexBuffers = &centerBuffer;
	state.input.sphereArray.vertexStrideInBytes = sizeof(float3);
	state.input.sphereArray.radiusBuffers = &radiusBuffer;
	state.input.sphereArray.radiusStrideInBytes = sizeof(float);
	state.input.sphereArray.numVertices = static_cast<unsigned int>(num);
	unsigned int flags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
	state.input.sphereArray.flags = flags;
	state.input.sphereArray.numSbtRecords = 1;
	state.input.sphereArray.sbtIndexOffsetBuffer = NULL;
	state.input.sphereArray.sbtIndexOffsetSizeInBytes = 0;
	state.input.sphereArray.sbtIndexOffsetStrideInBytes = 0;

	OptixAccelBufferSizes bufferSizes = {};
	OPTIX_CHECK(optixAccelComputeMemoryUsage(m_state.context, &accelOptions, &state.input, 1, &bufferSizes));

	CUdeviceptr tempBuffer = NULL, outputBuffer = NULL;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tempBuffer), bufferSizes.tempSizeInBytes));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputBuffer), bufferSizes.outputSizeInBytes));
	

	OPTIX_CHECK(optixAccelBuild(
		m_state.context,
		0,                  // CUDA stream
		&accelOptions,
		&state.input,
		1,                  // num build inputs
		tempBuffer,
		bufferSizes.tempSizeInBytes,
		outputBuffer,
		bufferSizes.outputSizeInBytes,
		&state.handle,
		nullptr,      // emitted property list
		0                   // num emitted properties
	));

	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(tempBuffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(outputBuffer)));
}

void Venusaur::RendererOptix::CreateModule(const char* filename)
{
	std::ifstream file(filename, std::ios::binary);
	std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});

	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef _DEBUG
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif // _DEBUG

	m_state.pipeline_compile_options.usesMotionBlur = false;
	m_state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	m_state.pipeline_compile_options.numPayloadValues = 2;
	m_state.pipeline_compile_options.numAttributeValues = 6;
	m_state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
	m_state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	m_state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

	char   log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixModuleCreate(
		m_state.context,
		&module_compile_options,
		&m_state.pipeline_compile_options,
		buffer.data(),
		buffer.size(),
		log,
		&sizeof_log,
		&m_state.ptx_module
	));
}

void Venusaur::RendererOptix::CreateProgramGroups()
{
	OptixProgramGroupDesc programGroupsDesc[5] = {};

	programGroupsDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	programGroupsDesc[0].raygen.module = m_state.ptx_module;
	programGroupsDesc[0].raygen.entryFunctionName = "__raygen__rg";

	programGroupsDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	programGroupsDesc[1].miss.module = m_state.ptx_module;
	programGroupsDesc[1].miss.entryFunctionName = "__miss__ms";

	programGroupsDesc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	programGroupsDesc[2].hitgroup.moduleCH = m_state.ptx_module;
	programGroupsDesc[2].hitgroup.entryFunctionNameCH = "__closesthit__lambertian";
	programGroupsDesc[2].hitgroup.moduleAH = nullptr;
	programGroupsDesc[2].hitgroup.entryFunctionNameAH = nullptr;
	programGroupsDesc[2].hitgroup.moduleIS = nullptr;
	programGroupsDesc[2].hitgroup.entryFunctionNameIS = nullptr;

	programGroupsDesc[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	programGroupsDesc[3].hitgroup.moduleCH = m_state.ptx_module;
	programGroupsDesc[3].hitgroup.entryFunctionNameCH = "__closesthit__metal";
	programGroupsDesc[3].hitgroup.moduleAH = nullptr;
	programGroupsDesc[3].hitgroup.entryFunctionNameAH = nullptr;
	programGroupsDesc[3].hitgroup.moduleIS = nullptr;
	programGroupsDesc[3].hitgroup.entryFunctionNameIS = nullptr;

	programGroupsDesc[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	programGroupsDesc[4].hitgroup.moduleCH = m_state.ptx_module;
	programGroupsDesc[4].hitgroup.entryFunctionNameCH = "__closesthit__dielectric";
	programGroupsDesc[4].hitgroup.moduleAH = nullptr;
	programGroupsDesc[4].hitgroup.entryFunctionNameAH = nullptr;
	programGroupsDesc[4].hitgroup.moduleIS = nullptr;
	programGroupsDesc[4].hitgroup.entryFunctionNameIS = nullptr;

	char   log[2048];
	size_t sizeof_log = sizeof(log);
	OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
	OPTIX_CHECK_LOG(optixProgramGroupCreate(
		m_state.context,
		programGroupsDesc,
		5,   // num program groups
		&program_group_options,
		log,
		&sizeof_log,
		m_state.programGroups
	));
}

void Venusaur::RendererOptix::CreatePipeline()
{
	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth = m_maxTraceDepth;

	char   log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixPipelineCreate(
		m_state.context,
		&m_state.pipeline_compile_options,
		&pipeline_link_options,
		m_state.programGroups,
		5,
		log,
		&sizeof_log,
		&m_state.pipeline
	));

	OptixStackSizes stack_sizes = {};
	for (auto& prog_group : m_state.programGroups)
	{
		OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, m_state.pipeline));
	}

	uint32_t direct_callable_stack_size_from_traversal;
	uint32_t direct_callable_stack_size_from_state;
	uint32_t continuation_stack_size;
	OPTIX_CHECK(optixUtilComputeStackSizes(
		&stack_sizes,
		m_maxTraceDepth,
		0,  // maxCCDepth
		0,  // maxDCDEpth
		&direct_callable_stack_size_from_traversal,
		&direct_callable_stack_size_from_state,
		&continuation_stack_size
	));

	OPTIX_CHECK(optixPipelineSetStackSize(
		m_state.pipeline,
		direct_callable_stack_size_from_traversal,
		direct_callable_stack_size_from_state,
		continuation_stack_size,
		1  // maxTraversableDepth
	));
}

void Venusaur::RendererOptix::CreateSBT()
{
	CUdeviceptr  raygen_record;
	RayGenSbtRecord rg_sbt;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), sizeof(RayGenSbtRecord)));
	OPTIX_CHECK(optixSbtRecordPackHeader(m_state.programGroups[0], &rg_sbt));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(raygen_record),
		&rg_sbt,
		sizeof(RayGenSbtRecord),
		cudaMemcpyHostToDevice
	));

	CUdeviceptr miss_record;
	MissSbtRecord ms_sbt;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), sizeof(MissSbtRecord)));
	OPTIX_CHECK(optixSbtRecordPackHeader(m_state.programGroups[1], &ms_sbt));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(miss_record),
		&ms_sbt,
		sizeof(MissSbtRecord),
		cudaMemcpyHostToDevice
	));

	CUdeviceptr hitgroup_records;
	size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_records), hitgroup_record_size * m_scene.getObjectsRef().size()));
	std::vector<HitGroupSbtRecord> hg_stbs(m_scene.getObjectsRef().size());
	for (size_t i = 0; i < m_scene.getObjectsRef().size(); ++i)
	{
		switch (m_scene.getObjectsRef()[i].type)
		{
		case Scene::MaterialType::Lambertian:
			hg_stbs[i].data = m_scene.getObjectMaterial(i);
			optixSbtRecordPackHeader(m_state.programGroups[2], &hg_stbs[i]);
			break;
		case Scene::MaterialType::Metal:
			hg_stbs[i].data = m_scene.getObjectMaterial(i);
			optixSbtRecordPackHeader(m_state.programGroups[3], &hg_stbs[i]);
			break;
		case Scene::MaterialType::Dielectric:
			hg_stbs[i].data = m_scene.getObjectMaterial(i);
			optixSbtRecordPackHeader(m_state.programGroups[4], &hg_stbs[i]);
		}
	}

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(hitgroup_records),
		hg_stbs.data(),
		hitgroup_record_size * m_scene.getObjectsRef().size(),
		cudaMemcpyHostToDevice
	));

	m_state.sbt.raygenRecord = raygen_record;
	m_state.sbt.missRecordBase = miss_record;
	m_state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
	m_state.sbt.missRecordCount = 1;
	m_state.sbt.hitgroupRecordBase = hitgroup_records;
	m_state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
	m_state.sbt.hitgroupRecordCount = static_cast<unsigned int>(m_scene.getObjectsRef().size());
}

uchar4* Venusaur::RendererOptix::MapResource()
{
	CUDA_CHECK(cudaSetDevice(0));
	CUDA_CHECK(cudaGraphicsMapResources(1, &m_pbo, m_state.stream));
	uchar4* cudaPtr;
	size_t size = 0;
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&cudaPtr), &size, m_pbo));
	return cudaPtr;
}

void Venusaur::RendererOptix::UnmapResource()
{
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_pbo, m_state.stream));
}
