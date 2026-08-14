#pragma once

#include <cstdint>
#include <memory>

#include <venusaur/gpu_resources.hpp>
#include <venusaur/result.hpp>

namespace venusaur {
class RenderTarget {
public:
    struct Mapping {
        uchar4* image = nullptr;
        ScopedGraphicsMap guard;
    };

    [[nodiscard]] static Result<std::shared_ptr<RenderTarget>> create(uint32_t width, uint32_t height);

    [[nodiscard]] Result<Mapping> map(CUstream stream);
    [[nodiscard]] Result<void> unmap(Mapping&& mapping);

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

private:
    RenderTarget() = default;

    uint32_t m_width = 0;
    uint32_t m_height = 0;

    GlTexture m_texture;
    GlBuffer m_pixelBuffer;
    CudaGraphicsRegistration m_cudaRegistration;
};
} // namespace venusaur
