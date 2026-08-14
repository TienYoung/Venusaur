#pragma once

#include <functional>
#include <memory>
#include <span>

#include <cuda_runtime.h>

#include <optix.h>

#include <venusaur/gpu_resources.hpp>
#include <venusaur/render_target.hpp>
#include <venusaur/result.hpp>

namespace venusaur {
class RayTracer {
public:
    [[nodiscard]] static Result<std::shared_ptr<RayTracer>> create();

    CUstream getCudaStream() const { return m_stream.get(); }
    OptixDeviceContext getOptixContext() const { return m_context.get(); }
    OptixPipeline getPipeline() const { return m_pipeline.get(); }

    [[nodiscard]] Result<void> allocateParams(std::size_t size);

    [[nodiscard]] Result<void> setupPipeline(const OptixPipelineCompileOptions& compile_options,
                                             const OptixPipelineLinkOptions& link_options,
                                             std::span<OptixProgramGroup> program_group);

    [[nodiscard]] Result<OptixTraversableHandle>
    createAccelBuffer(const OptixAccelBuildOptions& accel_build_options, const OptixBuildInput& build_input);

    void setupShaderBindingTable(OptixShaderBindingTable sbt,
                                 CudaDeviceBuffer raygenRecord,
                                 CudaDeviceBuffer missRecords,
                                 CudaDeviceBuffer hitgroupRecords);

    [[nodiscard]] Result<void> render(std::shared_ptr<RenderTarget> render_target);

    using SetupParamsCallback =
        std::function<Result<std::span<const std::byte>>(uchar4* image, uint32_t width, uint32_t height)>;
    void setRenderCallback(SetupParamsCallback callback) { m_setupParams = std::move(callback); }

    template <typename T> struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    template <> struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord<void> {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

private:
    RayTracer() = default;
    [[nodiscard]] Result<void> initialize();

    CudaStream m_stream;
    OptixContext m_context;
    OptixPipelineHandle m_pipeline;

    CudaDeviceBuffer m_accelBuffer;
    CudaDeviceBuffer m_params;
    std::size_t m_paramsCapacity = 0;

    CudaDeviceBuffer m_raygenRecord;
    CudaDeviceBuffer m_missRecords;
    CudaDeviceBuffer m_hitgroupRecords;

    OptixShaderBindingTable m_sbt = {};

    SetupParamsCallback m_setupParams;
};
} // namespace venusaur
