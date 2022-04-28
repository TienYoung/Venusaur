#include "Renderer.h"

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.raygenRecord)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.missRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.sbt.hitgroupRecordBase)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_gas_output_buffer)));
	CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_state.d_params)));

	OPTIX_CHECK(optixPipelineDestroy(m_state.pipeline));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.raygen_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.miss_prog_group));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.lambertian_hit_group));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.metal_hit_group));
	OPTIX_CHECK(optixProgramGroupDestroy(m_state.dielectric_hit_group));
	OPTIX_CHECK(optixModuleDestroy(m_state.ptx_module));
	OPTIX_CHECK(optixDeviceContextDestroy(m_state.context));
}

void Renderer::CreateContext()
{
	// Initialize CUDA
	CUDA_CHECK(cudaFree(0));

	OptixDeviceContext context;
	CUcontext cuCtx = 0;  // zero means take the current context
	OPTIX_CHECK(optixInit());
	OptixDeviceContextOptions options = {};
	options.logCallbackFunction       = &context_log_cb;
	options.logCallbackLevel          = 4;
	OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));

	m_state.context = context;
}

void Renderer::BuildAccelerationStructures(const Scene& scene)
{
	OptixAccelBuildOptions accel_options = {};
	accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
	accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

	// AABB build input
	const size_t aabbs_size_in_bytes = scene.m_aabbs.size() * sizeof(OptixAabb);
	CUdeviceptr d_aabb_buffer;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), aabbs_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_aabb_buffer),
		scene.m_aabbs.data(),
		aabbs_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	const size_t obj_indices_size_in_bytes = scene.m_indices.size() * sizeof(uint32_t);
	CUdeviceptr  d_obj_indices = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_obj_indices), obj_indices_size_in_bytes));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(d_obj_indices),
		scene.m_indices.data(),
		obj_indices_size_in_bytes,
		cudaMemcpyHostToDevice
	));

	OptixBuildInput aabb_input                    = {};
	aabb_input.type                               = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
	aabb_input.customPrimitiveArray.aabbBuffers   = &d_aabb_buffer;
	aabb_input.customPrimitiveArray.numPrimitives = scene.m_spheres.size();

	unsigned int* aabb_input_flags = new unsigned int[scene.m_spheres.size()];
	for (size_t i = 0; i < scene.m_spheres.size(); ++i)
	{
		aabb_input_flags[i] = OPTIX_GEOMETRY_FLAG_NONE;
	}
	aabb_input.customPrimitiveArray.flags = aabb_input_flags;
	aabb_input.customPrimitiveArray.numSbtRecords = scene.m_spheres.size();
	aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = d_obj_indices;
	aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);

	OptixAccelBufferSizes gas_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage(m_state.context, &accel_options, &aabb_input, 1, &gas_buffer_sizes));
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
	emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result             = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

	OPTIX_CHECK(optixAccelBuild(
		m_state.context,
		0,                  // CUDA stream
		&accel_options,
		&aabb_input,
		1,                  // num build inputs
		d_temp_buffer_gas,
		gas_buffer_sizes.tempSizeInBytes,
		d_buffer_temp_output_gas_and_compacted_size,
		gas_buffer_sizes.outputSizeInBytes,
		&m_state.gas_handle,
		&emitProperty,      // emitted property list
		1                   // num emitted properties
	));

	CUDA_CHECK(cudaFree((void*)d_temp_buffer_gas));
	CUDA_CHECK(cudaFree((void*)d_aabb_buffer));
	CUDA_CHECK(cudaFree((void*)d_obj_indices));
	delete[] aabb_input_flags;

	size_t compacted_gas_size;
	CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

	if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
	{
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_state.d_gas_output_buffer), compacted_gas_size));

		// use handle as input and output
		OPTIX_CHECK(optixAccelCompact(m_state.context, 0, m_state.gas_handle, m_state.d_gas_output_buffer, compacted_gas_size, &m_state.gas_handle));

		CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
	}
	else
	{
		m_state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
	}
}

void Renderer::CreateModule(const std::string& ptxSource)
{
	OptixModuleCompileOptions module_compile_options = {};
	module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
	module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

	m_state.pipeline_compile_options.usesMotionBlur = false;
	m_state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	m_state.pipeline_compile_options.numPayloadValues = 2;
	m_state.pipeline_compile_options.numAttributeValues = 6;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
	m_state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
	m_state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif // DEBUG
	m_state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

	//std::string currentPath(_getcwd(NULL, 0));
	//std::string filename = currentPath + "/../Binaries/x64/Debug/Hello.ptx";
	////std::string filename = currentPath + "./Hello.ptx";

	//size_t inputSize = 0;
	//std::fstream file(filename);
	//std::string source(std::istreambuf_iterator<char>(file), {});
	//const char* input = source.c_str();
	//inputSize = source.size();

	char   log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
		m_state.context,
		&module_compile_options,
		&m_state.pipeline_compile_options,
		ptxSource.c_str(),
		ptxSource.size(),
		log,
		&sizeof_log,
		&m_state.ptx_module
	));
}

void Renderer::CreateProgramGroups()
{
	OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

	char   log[2048];
	size_t sizeof_log = sizeof(log);
	{
		OptixProgramGroupDesc raygen_prog_group_desc = {}; 
		raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
		raygen_prog_group_desc.raygen.module = m_state.ptx_module;
		raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context,
			&raygen_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.raygen_prog_group
		));
	}

	{
		OptixProgramGroupDesc miss_prog_group_desc  = {};
		miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
		miss_prog_group_desc.miss.module            = m_state.ptx_module;
		miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context,
			&miss_prog_group_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.miss_prog_group
		));
	}

	{
		OptixProgramGroupDesc hitgroup_prog_group_lambertian_desc        = {};
		hitgroup_prog_group_lambertian_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_lambertian_desc.hitgroup.moduleCH            = m_state.ptx_module;
		hitgroup_prog_group_lambertian_desc.hitgroup.entryFunctionNameCH = "__closesthit__lambertian";
		hitgroup_prog_group_lambertian_desc.hitgroup.moduleAH            = nullptr;
		hitgroup_prog_group_lambertian_desc.hitgroup.entryFunctionNameAH = nullptr;
		hitgroup_prog_group_lambertian_desc.hitgroup.moduleIS            = m_state.ptx_module;
		hitgroup_prog_group_lambertian_desc.hitgroup.entryFunctionNameIS = "__intersection__hit_sphere";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context,
			&hitgroup_prog_group_lambertian_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.lambertian_hit_group
		));

		OptixProgramGroupDesc hitgroup_prog_group_metal_desc        = {};
		hitgroup_prog_group_metal_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_metal_desc.hitgroup.moduleCH            = m_state.ptx_module;
		hitgroup_prog_group_metal_desc.hitgroup.entryFunctionNameCH = "__closesthit__metal";
		hitgroup_prog_group_metal_desc.hitgroup.moduleAH            = nullptr;
		hitgroup_prog_group_metal_desc.hitgroup.entryFunctionNameAH = nullptr;
		hitgroup_prog_group_metal_desc.hitgroup.moduleIS            = m_state.ptx_module;
		hitgroup_prog_group_metal_desc.hitgroup.entryFunctionNameIS = "__intersection__hit_sphere";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context,
			&hitgroup_prog_group_metal_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.metal_hit_group
		));

		OptixProgramGroupDesc hitgroup_prog_group_dielectric_desc        = {};
		hitgroup_prog_group_dielectric_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
		hitgroup_prog_group_dielectric_desc.hitgroup.moduleCH            = m_state.ptx_module;
		hitgroup_prog_group_dielectric_desc.hitgroup.entryFunctionNameCH = "__closesthit__dielectric";
		hitgroup_prog_group_dielectric_desc.hitgroup.moduleAH            = nullptr;
		hitgroup_prog_group_dielectric_desc.hitgroup.entryFunctionNameAH = nullptr;
		hitgroup_prog_group_dielectric_desc.hitgroup.moduleIS            = m_state.ptx_module;
		hitgroup_prog_group_dielectric_desc.hitgroup.entryFunctionNameIS = "__intersection__hit_sphere";
		sizeof_log = sizeof(log);
		OPTIX_CHECK_LOG(optixProgramGroupCreate(
			m_state.context,
			&hitgroup_prog_group_dielectric_desc,
			1,   // num program groups
			&program_group_options,
			log,
			&sizeof_log,
			&m_state.dielectric_hit_group
		));
	}
}

void Renderer::CreatePipeline()
{
	OptixProgramGroup program_groups[] = { 
		m_state.raygen_prog_group, 
		m_state.miss_prog_group, 
		m_state.lambertian_hit_group, 
		m_state.metal_hit_group, 
		m_state.dielectric_hit_group 
	};

	OptixPipelineLinkOptions pipeline_link_options = {};
	pipeline_link_options.maxTraceDepth            = m_maxTraceDepth;
	pipeline_link_options.debugLevel               = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
	
	char   log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK_LOG(optixPipelineCreate(
		m_state.context,
		&m_state.pipeline_compile_options,
		&pipeline_link_options,
		program_groups,
		sizeof(program_groups) / sizeof(program_groups[0]),
		log,
		&sizeof_log,
		&m_state.pipeline
	));

	OptixStackSizes stack_sizes = {};
	for (auto& prog_group : program_groups)
	{
		OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
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

void Renderer::CreateSBT(const Scene& scene)
{
	CUdeviceptr  raygen_record;
	const size_t raygen_record_size = sizeof(RayGenSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
	RayGenSbtRecord rg_sbt;
	OPTIX_CHECK(optixSbtRecordPackHeader(m_state.raygen_prog_group, &rg_sbt));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(raygen_record),
		&rg_sbt,
		raygen_record_size,
		cudaMemcpyHostToDevice
	));

	CUdeviceptr miss_record;
	size_t      miss_record_size = sizeof(MissSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
	MissSbtRecord ms_sbt;
	OPTIX_CHECK(optixSbtRecordPackHeader(m_state.miss_prog_group, &ms_sbt));
	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(miss_record),
		&ms_sbt,
		miss_record_size,
		cudaMemcpyHostToDevice
	));

	CUdeviceptr hitgroup_records;
	size_t      hitgroup_record_size = sizeof(HitGroupSbtRecord);
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_records), hitgroup_record_size * scene.m_spheres.size()));
	std::vector<HitGroupSbtRecord> hg_stbs(scene.m_spheres.size());
	for (size_t i = 0; i < scene.m_spheres.size(); ++i)
	{
		hg_stbs[i].data.center = scene.m_spheres[i].GetCenter();
		hg_stbs[i].data.radius = scene.m_spheres[i].GetRadius();

		switch (scene.m_spheres[i].GetMaterial().GetType())
		{
		case Material::Lambertian:
			hg_stbs[i].data.mat.albedo = scene.m_spheres[i].GetMaterial().GetAlbedo();
			optixSbtRecordPackHeader(m_state.lambertian_hit_group, &hg_stbs[i]);
			break;
		case Material::Metal:
			hg_stbs[i].data.mat.albedo = scene.m_spheres[i].GetMaterial().GetAlbedo();
			hg_stbs[i].data.mat.fuzz = scene.m_spheres[i].GetMaterial().GetFuzz();
			optixSbtRecordPackHeader(m_state.metal_hit_group, &hg_stbs[i]);
			break;
		case Material::Dielectric:
			hg_stbs[i].data.mat.ir = scene.m_spheres[i].GetMaterial().GetIR();
			optixSbtRecordPackHeader(m_state.dielectric_hit_group, &hg_stbs[i]);
		default:
			break;
		}
	}

	CUDA_CHECK(cudaMemcpy(
		reinterpret_cast<void*>(hitgroup_records),
		hg_stbs.data(),
		hitgroup_record_size * scene.m_spheres.size(),
		cudaMemcpyHostToDevice
	));

	m_state.sbt.raygenRecord                = raygen_record;
	m_state.sbt.missRecordBase              = miss_record;
	m_state.sbt.missRecordStrideInBytes     = sizeof(MissSbtRecord);
	m_state.sbt.missRecordCount             = 1; 
	m_state.sbt.hitgroupRecordBase          = hitgroup_records;
	m_state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
	m_state.sbt.hitgroupRecordCount         = scene.m_spheres.size();
}

void Renderer::Launch()
{
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
	params.subframe_index = sub_index;
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

