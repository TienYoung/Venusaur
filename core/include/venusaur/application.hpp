#pragma once

#include <memory>

struct GLFWwindow;

namespace venusaur {
class RayTracer;
class RenderTarget;

class Application final {
public:
    Application(int width, int height);
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;

    [[nodiscard]] std::shared_ptr<RenderTarget> GetOutputBuffer() const;
    void SetRenderer(std::shared_ptr<RayTracer> renderer);

    void run();

private:
    struct State;

    std::unique_ptr<State> m_state;
    int m_width = 256;
    int m_height = 256;

    bool m_showUi = false;

    static void glfwErrorCallback(int error, const char* description);
    static void glfwKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void glfwResizeCallback(GLFWwindow* window, int width, int height);

    void onResize(int width, int height) {
        m_width = width;
        m_height = height;
    }
};
} // namespace venusaur
