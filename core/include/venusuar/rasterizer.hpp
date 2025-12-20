#pragma once

#include <GL/gl3w.h>

namespace venusaur {
class Rasterizer {
public:
    Rasterizer();
    ~Rasterizer();
    void render(GLuint width, GLuint height);

private:
    GLuint m_program = 0;
    GLuint m_vao = 0;
};
} // namespace venusaur