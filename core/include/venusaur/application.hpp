#pragma once

#include <memory>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <venusaur/rasterizer.hpp>
#include <venusaur/ray_tracer.hpp>
#include <venusaur/render_target.hpp>

namespace venusaur {
class Application {
public:
    Application(int width, int height);
    ~Application();

    std::shared_ptr<RenderTarget> GetOutputBuffer() const { return m_outputBuffer; }
    void SetRenderer(std::shared_ptr<RayTracer> renderer) { m_renderer = renderer; }

    void run();

private:
    GLFWwindow* m_window = nullptr;
    int m_width = 256;
    int m_height = 256;

    bool m_showUi = false;

    static void glfwErrorCallback(int error, const char* description);
    static void glfwKeyCallback(GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods);
    static void glfwResizeCallback(GLFWwindow* window, int width, int height);

    void onResize(int width, int height) {
        m_width = width;
        m_height = height;
    }

    pro::proxy<Renderable> m_renderer;
    std::shared_ptr<RenderTarget> m_outputBuffer = nullptr;
    std::shared_ptr<Rasterizer> m_rasterizer = nullptr;
};
} // namespace venusaur