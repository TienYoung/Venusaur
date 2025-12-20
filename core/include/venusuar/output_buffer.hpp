#pragma once

#include <cstdint>

#include <GL/gl3w.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

namespace venusaur {
class OutputBuffer {
public:
    OutputBuffer(uint32_t width, uint32_t height);
    ~OutputBuffer();

    uchar4* map(CUstream stream);
    void unmap(CUstream stream);

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }

private:
    uint32_t m_width = 0;
    uint32_t m_height = 0;

    GLuint m_tex = 0;
    GLuint m_pbo = 0;

    cudaGraphicsResource* m_gfxResource = nullptr;
};
} // namespace venusaur
