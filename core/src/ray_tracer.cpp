#include <venusaur/ray_tracer.hpp>

#include <format>
#include <string_view>
#include <utility>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <spdlog/spdlog.h>

#include <venusaur/exception.hpp>

namespace venusaur {
namespace {
void contextLogCallback(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    spdlog::level::level_enum logLevel = spdlog::level::debug;
    switch (level) {
    case 1: // fatal
        logLevel = spdlog::level::critical;
        break;
    case 2: // error
        logLevel = spdlog::level::err;
        break;
    case 3: // warning
        logLevel = spdlog::level::warn;
        break;
    case 4: // print / info
        logLevel = spdlog::level::info;
        break;
    default:
        break;
    }
    spdlog::log(logLevel, "[OptiX] [{}] {}", tag, message);
}
} // namespace

Result<std::shared_ptr<RayTracer>> RayTracer::create() {
    auto rayTracer = std::shared_ptr<RayTracer>(new RayTracer{});
    if (auto result = rayTracer->initialize(); !result) {
        return std::unexpected(std::move(result.error()));
    }
    return rayTracer;
}

Result<void> RayTracer::initialize() {
    if (auto result = checkCuda(cudaFree(nullptr), "initialize CUDA runtime"); !result) {
        return result;
    }

    auto stream = createCudaStream();
    if (!stream) {
        return std::unexpected(std::move(stream.error()));
    }
    m_stream = std::move(*stream);

    CUcontext cuCtx = 0; // zero means take the current context
    if (auto result = checkOptix(optixInit(), "optixInit"); !result) {
        return result;
    }

    OptixDeviceContextOptions options = {
        .logCallbackFunction = &contextLogCallback,
        .logCallbackData = nullptr,
        .logCallbackLevel = 4,
#ifdef _DEBUG
        // This may incur significant performance cost and should only be done during development.
        .validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL,
#endif
    };

    OptixDeviceContext context = nullptr;
    const OptixResult contextResult = optixDeviceContextCreate(cuCtx, &options, &context);
    OptixContext contextOwner{context};
    if (auto result = checkOptix(contextResult, "optixDeviceContextCreate"); !result) {
        return result;
    }
    m_context = std::move(contextOwner);
    return {};
}

Result<void> RayTracer::allocateParams(std::size_t size) {
    auto params = allocateDeviceBuffer(size);
    if (!params) {
        return std::unexpected(std::move(params.error()));
    }
    m_params = std::move(*params);
    m_paramsCapacity = size;
    return {};
}

Result<void> RayTracer::setupPipeline(const OptixPipelineCompileOptions& compile_options,
                                      const OptixPipelineLinkOptions& link_options,
                                      std::span<OptixProgramGroup> program_group) {

    char log[2048];
    size_t log_length = sizeof(log);
    OptixPipeline pipeline = nullptr;
    const OptixResult createResult = optixPipelineCreate(m_context.get(),
                                                         &compile_options,
                                                         &link_options,
                                                         program_group.data(),
                                                         program_group.size(),
                                                         log,
                                                         &log_length,
                                                         &pipeline);
    OptixPipelineHandle pipelineOwner{pipeline};
    if (auto result = checkOptix(createResult, "optixPipelineCreate", std::string_view(log, log_length)); !result) {
        return result;
    }

    OptixStackSizes stackSizes = {};
    for (auto& progGroup : program_group) {
        if (auto result = checkOptix(optixUtilAccumulateStackSizes(progGroup, &stackSizes, pipeline),
                                     "optixUtilAccumulateStackSizes");
            !result) {
            return result;
        }
    }

    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;
    if (auto result = checkOptix(optixUtilComputeStackSizes(&stackSizes,
                                                            link_options.maxTraceDepth,
                                                            0,
                                                            0,
                                                            &directCallableStackSizeFromTraversal,
                                                            &directCallableStackSizeFromState,
                                                            &continuationStackSize),
                                 "optixUtilComputeStackSizes");
        !result) {
        return result;
    }

    if (auto result = checkOptix(optixPipelineSetStackSize(pipeline,
                                                           directCallableStackSizeFromTraversal,
                                                           directCallableStackSizeFromState,
                                                           continuationStackSize,
                                                           1),
                                 "optixPipelineSetStackSize");
        !result) {
        return result;
    }

    m_pipeline = std::move(pipelineOwner);
    return {};
}

Result<OptixTraversableHandle> RayTracer::createAccelBuffer(const OptixAccelBuildOptions& accel_build_options,
                                                           const OptixBuildInput& build_input) {
    OptixAccelBufferSizes accel_buffer_sizes = {};
    if (auto result = checkOptix(optixAccelComputeMemoryUsage(
                                     m_context.get(), &accel_build_options, &build_input, 1, &accel_buffer_sizes),
                                 "optixAccelComputeMemoryUsage");
        !result) {
        return std::unexpected(std::move(result.error()));
    }

    auto tempBuffer = allocateDeviceBuffer(accel_buffer_sizes.tempSizeInBytes);
    if (!tempBuffer) {
        return std::unexpected(std::move(tempBuffer.error()));
    }

    auto accelBuffer = allocateDeviceBuffer(accel_buffer_sizes.outputSizeInBytes);
    if (!accelBuffer) {
        return std::unexpected(std::move(accelBuffer.error()));
    }

    OptixTraversableHandle traversable_handle = 0;
    if (auto result = checkOptix(optixAccelBuild(m_context.get(),
                                                 m_stream.get(),
                                                 &accel_build_options,
                                                 &build_input,
                                                 1,
                                                 tempBuffer->get(),
                                                 accel_buffer_sizes.tempSizeInBytes,
                                                 accelBuffer->get(),
                                                 accel_buffer_sizes.outputSizeInBytes,
                                                 &traversable_handle,
                                                 nullptr,
                                                 0),
                                 "optixAccelBuild");
        !result) {
        return std::unexpected(std::move(result.error()));
    }

    m_accelBuffer = std::move(*accelBuffer);
    return traversable_handle;
}

void RayTracer::setupShaderBindingTable(OptixShaderBindingTable sbt,
                                        CudaDeviceBuffer raygenRecord,
                                        CudaDeviceBuffer missRecords,
                                        CudaDeviceBuffer hitgroupRecords) {
    m_raygenRecord = std::move(raygenRecord);
    m_missRecords = std::move(missRecords);
    m_hitgroupRecords = std::move(hitgroupRecords);

    sbt.raygenRecord = m_raygenRecord.get();
    sbt.missRecordBase = m_missRecords.get();
    sbt.hitgroupRecordBase = m_hitgroupRecords.get();
    m_sbt = sbt;
}

Result<void> RayTracer::render(std::shared_ptr<RenderTarget> render_target) {
    if (!render_target) {
        return std::unexpected(Error{
            .domain = ErrorDomain::application,
            .operation = "RayTracer::render",
            .message = "Render target is null",
        });
    }
    if (!m_pipeline || !m_params || m_sbt.raygenRecord == 0) {
        return std::unexpected(Error{
            .domain = ErrorDomain::application,
            .operation = "RayTracer::render",
            .message = "Pipeline, launch parameters, or shader binding table is not configured",
        });
    }
    if (!m_setupParams) {
        return std::unexpected(Error{
            .domain = ErrorDomain::application,
            .operation = "RayTracer::render",
            .message = "Render callback is not configured",
        });
    }

    auto mapping = render_target->map(m_stream.get());
    if (!mapping) {
        return std::unexpected(std::move(mapping.error()));
    }

    auto params = m_setupParams(mapping->image, render_target->getWidth(), render_target->getHeight());
    if (!params) {
        return std::unexpected(std::move(params.error()));
    }
    if (params->size() > m_paramsCapacity) {
        return std::unexpected(Error{
            .domain = ErrorDomain::application,
            .operation = "RayTracer::render",
            .message = std::format("Launch parameter size {} exceeds allocation capacity {}",
                                   params->size(),
                                   m_paramsCapacity),
        });
    }

    if (auto result = checkCuda(cudaMemcpy(reinterpret_cast<void*>(m_params.get()),
                                           params->data(),
                                           params->size(),
                                           cudaMemcpyHostToDevice),
                                "copy launch parameters");
        !result) {
        return result;
    }

    if (auto result = checkOptix(optixLaunch(m_pipeline.get(),
                                             m_stream.get(),
                                             m_params.get(),
                                             params->size(),
                                             &m_sbt,
                                             render_target->getWidth(),
                                             render_target->getHeight(),
                                             1),
                                 "optixLaunch");
        !result) {
        return result;
    }

    return render_target->unmap(std::move(*mapping));
}

} // namespace venusaur
