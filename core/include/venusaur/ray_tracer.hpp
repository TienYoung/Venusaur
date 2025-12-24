#pragma once

#include <functional>
#include <memory>
#include <span>

#include <proxy/proxy.h>

#include <cuda_runtime.h>

#include <optix.h>

#include <venusaur/exception.hpp>
#include <venusaur/render_target.hpp>

namespace venusaur {
PRO_DEF_MEM_DISPATCH(RENDER, render);

struct Renderable : ::pro::facade_builder ::add_convention<RENDER, void(std::shared_ptr<RenderTarget>)>::build {};

class RayTracer {
public:
    RayTracer();
    ~RayTracer();

    CUstream getCudaStream() const { return m_stream; }
    OptixDeviceContext getOptixContext() const { return m_context; }
    OptixPipeline getPipeline() const { return m_pipeline; }

    void mallocParamsOnDevice(size_t size) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), size)); }

    void setupPipeline(const OptixPipelineCompileOptions& compile_options,
                       const OptixPipelineLinkOptions& link_options,
                       std::span<OptixProgramGroup> program_group);

    [[nodiscard]] OptixTraversableHandle createAccelBuffer(const OptixAccelBuildOptions& accel_build_options,
                                                           const OptixBuildInput& build_input);

    void setupShaderBindingTable(OptixShaderBindingTable&& sbt) { m_sbt = sbt; };

    void render(std::shared_ptr<RenderTarget> render_target);

    using SetupParamsCallback =
        std::function<std::span<const std::byte>(uchar4* image, uint32_t width, uint32_t height)>;
    void setRenderCallback(SetupParamsCallback callback) { m_setupParams = std::move(callback); }

    template <typename T> struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    template <> struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord<void> {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

protected:
    CUstream m_stream = nullptr;

    OptixDeviceContext m_context = nullptr;

    OptixPipeline m_pipeline = nullptr;

    CUdeviceptr d_accelBuffer = NULL;

    CUdeviceptr d_params = NULL;

    OptixShaderBindingTable m_sbt = {};

    SetupParamsCallback m_setupParams;
};
} // namespace venusaur