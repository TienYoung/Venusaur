#include <venusaur/application.hpp>

#include <GL/gl3w.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <spdlog/spdlog.h>

namespace venusaur {
void Application::glfwErrorCallback(int error, const char* description) {
    spdlog::error("[GLFW][Error {}] {}", error, description);
}

void Application::glfwKeyCallback(
    GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/) {
    if (action == GLFW_RELEASE) {
        switch (key) {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, true);
            break;
        case GLFW_KEY_TAB:
            if (glfwGetWindowAttrib(window, GLFW_DECORATED) == GLFW_TRUE) {
                glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_FALSE);
            } else {
                glfwSetWindowAttrib(window, GLFW_DECORATED, GLFW_TRUE);
            }
            break;
        case GLFW_KEY_F1:
            auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
            app->m_showUi = !app->m_showUi;
            break;
        }
    }
}

void Application::glfwResizeCallback(GLFWwindow* window, int width, int height) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app) {
        app->onResize(width, height);
    }
}

Application::Application(int width, int height) : m_width(width), m_height(height) {
    // Init glfw.
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        throw std::runtime_error("Failed to init GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_DEBUG, true);

    m_window = glfwCreateWindow(m_width, m_height, "Venusaur", nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create a GLFW window!");
    }

    glfwSetKeyCallback(m_window, glfwKeyCallback);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetWindowSizeLimits(m_window, m_width, m_height, GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwSetWindowAspectRatio(m_window, m_width, m_height);
    glfwSetWindowSizeCallback(m_window, glfwResizeCallback);
    glfwMakeContextCurrent(m_window);

    glfwSwapInterval(1);

    m_outputBuffer = std::make_shared<RenderTarget>(m_width, m_height);
    m_rasterizer = std::make_shared<Rasterizer>();

    // Init ImGui.
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 460 core");

#ifdef _WIN32
    ImFont* font = io.Fonts->AddFontFromFileTTF(
        R"(c:\Windows\Fonts\SegoeUI.ttf)", 18.0f, nullptr, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
    IM_ASSERT(font != nullptr);
#endif
}

Application::~Application() {
    m_outputBuffer.reset();
    m_renderer.reset();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void Application::run() {
    while (!glfwWindowShouldClose(m_window)) {
        auto startPoint = std::chrono::high_resolution_clock::now();

        m_renderer->render(m_outputBuffer);
        m_rasterizer->render(m_width, m_height);

        auto endPoint = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endPoint - startPoint);

        glfwSetWindowTitle(m_window, std::format("Venusaur - {}ms", duration.count()).c_str());

        if (m_showUi) {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::Begin("Debugging", nullptr);
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.6f, 0.1f, 1.0f));

            ImGui::Text("OpenGL:");
            ImGui::Indent();
            ImGui::Text("Time:\t%lldms", duration.count());
            ImGui::Text("FPS:\t%lld", 1000 / (duration.count() + 1));
            ImGui::Unindent();

            ImGui::PopStyleColor();
            ImGui::End();
            ImGui::EndFrame();

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}
} // namespace venusaur