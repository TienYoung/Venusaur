#include <venusuar/output_buffer.hpp>

#include <venusuar/exception.hpp>

namespace venusaur {
OutputBuffer::OutputBuffer(uint32_t width, uint32_t height) : m_width(width), m_height(height) {
    if (gl3wInit()) {
        throw std::runtime_error("Failed to initialize GL");
    }

    // Create Texture
    glCreateTextures(GL_TEXTURE_2D, 1, &m_tex);
    glTextureStorage2D(m_tex, 1, GL_RGBA8, m_width, m_height);
    glTextureParameteri(m_tex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTextureParameteri(m_tex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(m_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTextureUnit(0, m_tex);

    // Create PBO
    glCreateBuffers(1, &m_pbo);
    glNamedBufferData(m_pbo, sizeof(uint8_t) * 4 * m_width * m_height, nullptr, GL_STREAM_DRAW);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_gfxResource, m_pbo, cudaGraphicsMapFlagsWriteDiscard));
}

OutputBuffer::~OutputBuffer() {
    CUDA_CHECK(cudaGraphicsUnregisterResource(m_gfxResource));
}

uchar4* OutputBuffer::map(CUstream stream) {
    uchar4* image = nullptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_gfxResource, stream));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&image), nullptr, m_gfxResource));
    return image;
}

void OutputBuffer::unmap(CUstream stream) {
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_gfxResource, stream));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTextureSubImage2D(m_tex, 0, 0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
} // namespace venusaur
