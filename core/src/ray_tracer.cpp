#include <venusaur/ray_tracer.hpp>

#include <format>

#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <spdlog/spdlog.h>

#include <venusaur/exception.hpp>

namespace venusaur {
namespace {
void contextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
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

    const auto log_msg = std::format("[OptiX] [{}] {}", tag, message);
    switch (level) {
    case 1: // fatal
        spdlog::critical(log_msg);
        break;
    case 2: // error
        spdlog::error(log_msg);
        break;
    case 3: // warning
        spdlog::warn(log_msg);
        break;
    case 4: // print / info
        spdlog::info(log_msg);
        break;
    default: // others
        spdlog::debug(log_msg);
        break;
    }
}
} // namespace

RayTracer::RayTracer() {
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {
        .logCallbackFunction = &contextLogCallback,
        .logCallbackData = nullptr,
        .logCallbackLevel = 4,
#ifdef _DEBUG
        // This may incur significant performance cost and should only be done during development.
        .validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,
#endif
    };
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
}

RayTracer::~RayTracer() {
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_accelBuffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(m_sbt.hitgroupRecordBase)));

    OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
    OPTIX_CHECK(optixDeviceContextDestroy(m_context));
}

void RayTracer::setupPipeline(const OptixPipelineCompileOptions& compile_options,
                              const OptixPipelineLinkOptions& link_options,
                              std::span<OptixProgramGroup> program_group) {

    char log[2048];
    size_t log_length = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(m_context,
                                        &compile_options,
                                        &link_options,
                                        program_group.data(),
                                        program_group.size(),
                                        log,
                                        &log_length,
                                        &m_pipeline));

    OptixStackSizes stackSizes = {};
    for (auto& progGroup : program_group) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(progGroup, &stackSizes, m_pipeline));
    }

    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
                                           link_options.maxTraceDepth,
                                           0,
                                           0,
                                           &directCallableStackSizeFromTraversal,
                                           &directCallableStackSizeFromState,
                                           &continuationStackSize));

    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline, directCallableStackSizeFromTraversal, directCallableStackSizeFromState, continuationStackSize, 1));
}

OptixTraversableHandle RayTracer::createAccelBuffer(const OptixAccelBuildOptions& accel_build_options,
                                                    const OptixBuildInput& build_input) {
    OptixAccelBufferSizes accel_buffer_sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_build_options, &build_input, 1, &accel_buffer_sizes));

    CUdeviceptr d_gasTempBuffer = NULL;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasTempBuffer), accel_buffer_sizes.tempSizeInBytes));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_accelBuffer), accel_buffer_sizes.outputSizeInBytes));

    OptixTraversableHandle traversable_handle = 0;
    OPTIX_CHECK(optixAccelBuild(m_context,
                                m_stream,
                                &accel_build_options,
                                &build_input,
                                1,
                                d_gasTempBuffer,
                                accel_buffer_sizes.tempSizeInBytes,
                                d_accelBuffer,
                                accel_buffer_sizes.outputSizeInBytes,
                                &traversable_handle,
                                nullptr,
                                0));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gasTempBuffer)));

    return traversable_handle;
}

void RayTracer::render(std::shared_ptr<RenderTarget> render_target) {
    uchar4* image = render_target->map(m_stream);
    auto params = m_setupParams(image, render_target->getWidth(), render_target->getHeight());
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), params.data(), params.size(), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(m_pipeline,
                            m_stream,
                            d_params,
                            params.size(),
                            &m_sbt,
                            render_target->getWidth(),
                            render_target->getHeight(),
                            1));
    render_target->unmap(m_stream);
}

} // namespace venusaur