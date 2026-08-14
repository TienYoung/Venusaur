#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <utility>

#include <GL/gl3w.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <optix.h>

#include <venusaur/exception.hpp>
#include <venusaur/result.hpp>
#include <venusaur/unique_resource.hpp>

namespace venusaur {
struct GlTextureDeleter {
    void operator()(GLuint value) const noexcept { glDeleteTextures(1, &value); }
};

struct GlBufferDeleter {
    void operator()(GLuint value) const noexcept { glDeleteBuffers(1, &value); }
};

struct GlShaderDeleter {
    void operator()(GLuint value) const noexcept { glDeleteShader(value); }
};

struct GlProgramDeleter {
    void operator()(GLuint value) const noexcept { glDeleteProgram(value); }
};

struct GlVertexArrayDeleter {
    void operator()(GLuint value) const noexcept { glDeleteVertexArrays(1, &value); }
};

struct CudaMemoryDeleter {
    void operator()(CUdeviceptr value) const noexcept { CUDA_CHECK_NOTHROW(cudaFree(reinterpret_cast<void*>(value))); }
};

struct CudaStreamDeleter {
    void operator()(CUstream value) const noexcept { CUDA_CHECK_NOTHROW(cudaStreamDestroy(value)); }
};

struct CudaGraphicsResourceDeleter {
    void operator()(cudaGraphicsResource* value) const noexcept {
        CUDA_CHECK_NOTHROW(cudaGraphicsUnregisterResource(value));
    }
};

struct OptixDeviceContextDeleter {
    void operator()(OptixDeviceContext value) const noexcept { OPTIX_CHECK_NOTHROW(optixDeviceContextDestroy(value)); }
};

struct OptixPipelineDeleter {
    void operator()(OptixPipeline value) const noexcept { OPTIX_CHECK_NOTHROW(optixPipelineDestroy(value)); }
};

struct OptixModuleDeleter {
    void operator()(OptixModule value) const noexcept { OPTIX_CHECK_NOTHROW(optixModuleDestroy(value)); }
};

struct OptixProgramGroupDeleter {
    void operator()(OptixProgramGroup value) const noexcept { OPTIX_CHECK_NOTHROW(optixProgramGroupDestroy(value)); }
};

struct GraphicsMapHandle {
    cudaGraphicsResource* resource = nullptr;
    CUstream stream = nullptr;

    constexpr bool operator==(const GraphicsMapHandle&) const noexcept = default;
};

struct GraphicsMapDeleter {
    void operator()(GraphicsMapHandle value) const noexcept {
        CUDA_CHECK_NOTHROW(cudaGraphicsUnmapResources(1, &value.resource, value.stream));
    }
};

using GlTexture = UniqueResource<GLuint, 0, GlTextureDeleter>;
using GlBuffer = UniqueResource<GLuint, 0, GlBufferDeleter>;
using GlShader = UniqueResource<GLuint, 0, GlShaderDeleter>;
using GlProgram = UniqueResource<GLuint, 0, GlProgramDeleter>;
using GlVertexArray = UniqueResource<GLuint, 0, GlVertexArrayDeleter>;
using CudaDeviceBuffer = UniqueResource<CUdeviceptr, 0, CudaMemoryDeleter>;
using CudaStream = UniqueResource<CUstream, nullptr, CudaStreamDeleter>;
using CudaGraphicsRegistration = UniqueResource<cudaGraphicsResource*, nullptr, CudaGraphicsResourceDeleter>;
using OptixContext = UniqueResource<OptixDeviceContext, nullptr, OptixDeviceContextDeleter>;
using OptixPipelineHandle = UniqueResource<OptixPipeline, nullptr, OptixPipelineDeleter>;
using OptixModuleHandle = UniqueResource<OptixModule, nullptr, OptixModuleDeleter>;
using OptixProgramGroupHandle = UniqueResource<OptixProgramGroup, nullptr, OptixProgramGroupDeleter>;
using ScopedGraphicsMap = UniqueResource<GraphicsMapHandle, GraphicsMapHandle{}, GraphicsMapDeleter>;

[[nodiscard]] inline Error cudaError(cudaError_t code, std::string_view operation) {
    return Error{
        .domain = ErrorDomain::cuda,
        .code = static_cast<int>(code),
        .operation = std::string(operation),
        .message = cudaGetErrorString(code),
    };
}

[[nodiscard]] inline Error optixError(OptixResult code, std::string_view operation, std::string_view detail = {}) {
    std::string message = optixGetErrorString(code);
    if (!detail.empty()) {
        message.append("\n");
        message.append(detail);
    }
    return Error{
        .domain = ErrorDomain::optix,
        .code = static_cast<int>(code),
        .operation = std::string(operation),
        .message = std::move(message),
    };
}

[[nodiscard]] inline Error glError(GLenum code, std::string_view operation) {
    return Error{
        .domain = ErrorDomain::opengl,
        .code = static_cast<int>(code),
        .operation = std::string(operation),
        .message = sutil::getGLErrorString(code),
    };
}

[[nodiscard]] inline Result<void> checkCuda(cudaError_t code, std::string_view operation) {
    if (code != cudaSuccess) {
        return std::unexpected(cudaError(code, operation));
    }
    return {};
}

[[nodiscard]] inline Result<void>
checkOptix(OptixResult code, std::string_view operation, std::string_view detail = {}) {
    if (code != OPTIX_SUCCESS) {
        return std::unexpected(optixError(code, operation, detail));
    }
    return {};
}

[[nodiscard]] inline Result<void> checkGl(std::string_view operation) {
    const GLenum code = glGetError();
    if (code != GL_NO_ERROR) {
        return std::unexpected(glError(code, operation));
    }
    return {};
}

[[nodiscard]] inline Result<CudaDeviceBuffer> allocateDeviceBuffer(std::size_t size) {
    void* allocation = nullptr;
    const cudaError_t code = cudaMalloc(&allocation, size);
    CudaDeviceBuffer owner{reinterpret_cast<CUdeviceptr>(allocation)};
    if (code != cudaSuccess) {
        return std::unexpected(cudaError(code, "cudaMalloc"));
    }
    return owner;
}

[[nodiscard]] inline Result<CudaStream> createCudaStream() {
    CUstream stream = nullptr;
    const cudaError_t code = cudaStreamCreate(&stream);
    CudaStream owner{stream};
    if (code != cudaSuccess) {
        return std::unexpected(cudaError(code, "cudaStreamCreate"));
    }
    return owner;
}

[[nodiscard]] inline Result<CudaGraphicsRegistration> registerCudaGraphicsBuffer(GLuint buffer) {
    cudaGraphicsResource* resource = nullptr;
    const cudaError_t code = cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsWriteDiscard);
    CudaGraphicsRegistration owner{resource};
    if (code != cudaSuccess) {
        return std::unexpected(cudaError(code, "cudaGraphicsGLRegisterBuffer"));
    }
    return owner;
}
} // namespace venusaur
