#include <venusaur/render_target.hpp>

#include <utility>

namespace venusaur {
Result<std::shared_ptr<RenderTarget>> RenderTarget::create(uint32_t width, uint32_t height) {
    if (gl3wInit()) {
        return std::unexpected(Error{
            .domain = ErrorDomain::opengl,
            .operation = "gl3wInit",
            .message = "Failed to initialize OpenGL function loading",
        });
    }

    auto target = std::shared_ptr<RenderTarget>(new RenderTarget{});
    target->m_width = width;
    target->m_height = height;

    GLuint texture = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    target->m_texture.reset(texture);
    glTextureStorage2D(texture, 1, GL_RGBA8, width, height);
    glTextureParameteri(texture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(texture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTextureUnit(0, texture);
    if (auto result = checkGl("create render target texture"); !result) {
        return std::unexpected(std::move(result.error()));
    }

    GLuint pixelBuffer = 0;
    glCreateBuffers(1, &pixelBuffer);
    target->m_pixelBuffer.reset(pixelBuffer);
    glNamedBufferData(pixelBuffer, sizeof(uint8_t) * 4 * width * height, nullptr, GL_STREAM_DRAW);
    if (auto result = checkGl("create render target pixel buffer"); !result) {
        return std::unexpected(std::move(result.error()));
    }

    auto registration = registerCudaGraphicsBuffer(pixelBuffer);
    if (!registration) {
        return std::unexpected(std::move(registration.error()));
    }
    target->m_cudaRegistration = std::move(*registration);

    return target;
}

Result<RenderTarget::Mapping> RenderTarget::map(CUstream stream) {
    cudaGraphicsResource* resource = m_cudaRegistration.get();
    const cudaError_t mapCode = cudaGraphicsMapResources(1, &resource, stream);
    if (mapCode != cudaSuccess) {
        return std::unexpected(cudaError(mapCode, "cudaGraphicsMapResources"));
    }

    ScopedGraphicsMap guard{GraphicsMapHandle{.resource = resource, .stream = stream}};
    uchar4* image = nullptr;
    const cudaError_t pointerCode =
        cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&image), nullptr, resource);
    if (pointerCode != cudaSuccess) {
        return std::unexpected(cudaError(pointerCode, "cudaGraphicsResourceGetMappedPointer"));
    }

    return Mapping{.image = image, .guard = std::move(guard)};
}

Result<void> RenderTarget::unmap(Mapping&& mapping) {
    GraphicsMapHandle handle = mapping.guard.release();
    const cudaError_t unmapCode = cudaGraphicsUnmapResources(1, &handle.resource, handle.stream);
    if (unmapCode != cudaSuccess) {
        return std::unexpected(cudaError(unmapCode, "cudaGraphicsUnmapResources"));
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pixelBuffer.get());
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTextureSubImage2D(m_texture.get(), 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    return checkGl("upload render target texture");
}
} // namespace venusaur
