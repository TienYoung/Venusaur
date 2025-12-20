#pragma once

#include <memory>

#include <cuda_runtime.h>

#include <optix.h>

#include <venusuar/output_buffer.hpp>

namespace venusaur {
class RendererBase {
public:
    RendererBase(std::shared_ptr<OutputBuffer> outputBuffer, uint32_t maxTraceDepth);

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

    template <typename T> struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    template <> struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord<void> {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };

    virtual size_t UpdateParams() = 0;
};
} // namespace venusaur