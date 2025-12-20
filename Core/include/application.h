#pragma once

#include "rasterizer.h"
#include <memory>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace Venusaur
{
    class RendererBase;
    class OutputBuffer;
    class Rasterizer;
    
    class Application
    {
    public:
        Application(int width, int height);
        ~Application();

        inline std::shared_ptr<OutputBuffer> GetOutputBuffer() const { return m_outputBuffer; }
        inline void SetRenderer(std::shared_ptr<RendererBase> renderer) { m_renderer = renderer; }

        void Update();

        inline bool IsRunning() const { return glfwWindowShouldClose(m_window) == GLFW_FALSE; }
        inline void ResizeWindow(int width, int height) { m_width = width; m_height = height; }
        inline void ToggleUi() { m_showUi = !m_showUi; }
    
    private:
        GLFWwindow* m_window = nullptr;
        int m_width = 256;
        int m_height = 256;

        bool m_showUi = false;

        std::shared_ptr<RendererBase> m_renderer = nullptr;
        std::shared_ptr<OutputBuffer> m_outputBuffer = nullptr;
        std::shared_ptr<Rasterizer> m_rasterizer = nullptr;
    };
}