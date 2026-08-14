#pragma once

#include <memory>

#include <venusaur/gpu_resources.hpp>
#include <venusaur/result.hpp>

namespace venusaur {
class Rasterizer {
public:
    [[nodiscard]] static Result<std::shared_ptr<Rasterizer>> create();

    void render(GLuint width, GLuint height);

private:
    Rasterizer() = default;

    GlProgram m_program;
    GlVertexArray m_vertexArray;
};
} // namespace venusaur
