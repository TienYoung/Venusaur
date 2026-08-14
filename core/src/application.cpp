#include <venusaur/application.hpp>

#include <chrono>
#include <format>
#include <memory>
#include <stdexcept>
#include <utility>

#include <GL/gl3w.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <spdlog/spdlog.h>

#include <venusaur/rasterizer.hpp>
#include <venusaur/ray_tracer.hpp>
#include <venusaur/render_target.hpp>

namespace venusaur {
namespace {
class GlfwRuntime {
public:
    explicit GlfwRuntime(GLFWerrorfun errorCallback) {
        if (s_active) {
            throw std::logic_error("Only one Application can be active at a time");
        }

        glfwSetErrorCallback(errorCallback);
        if (glfwInit() != GLFW_TRUE) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
        s_active = true;
    }

    ~GlfwRuntime() noexcept {
        glfwTerminate();
        s_active = false;
    }

    GlfwRuntime(const GlfwRuntime&) = delete;
    GlfwRuntime& operator=(const GlfwRuntime&) = delete;

private:
    static inline bool s_active = false;
};

struct WindowDeleter {
    void operator()(GLFWwindow* window) const noexcept {
        if (window != nullptr) {
            glfwDestroyWindow(window);
        }
    }
};

using Window = std::unique_ptr<GLFWwindow, WindowDeleter>;

class ImGuiSession {
public:
    ImGuiSession() = default;

    ~ImGuiSession() noexcept {
        if (m_openglBackendInitialized) {
            ImGui_ImplOpenGL3_Shutdown();
        }
        if (m_glfwBackendInitialized) {
            ImGui_ImplGlfw_Shutdown();
        }
        if (m_contextCreated) {
            ImGui::DestroyContext();
        }
    }

    ImGuiSession(const ImGuiSession&) = delete;
    ImGuiSession& operator=(const ImGuiSession&) = delete;

    void initialize(GLFWwindow* window) {
        IMGUI_CHECKVERSION();
        if (ImGui::CreateContext() == nullptr) {
            throw std::runtime_error("Failed to create ImGui context");
        }
        m_contextCreated = true;

        ImGuiIO& io = ImGui::GetIO();
        ImGui::StyleColorsDark();

        if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
            throw std::runtime_error("Failed to initialize ImGui GLFW backend");
        }
        m_glfwBackendInitialized = true;

        if (!ImGui_ImplOpenGL3_Init("#version 460 core")) {
            throw std::runtime_error("Failed to initialize ImGui OpenGL backend");
        }
        m_openglBackendInitialized = true;

#ifdef _WIN32
        ImFont* font = io.Fonts->AddFontFromFileTTF(
            R"(c:\Windows\Fonts\SegoeUI.ttf)", 18.0f, nullptr, io.Fonts->GetGlyphRangesChineseSimplifiedCommon());
        if (font == nullptr) {
            throw std::runtime_error("Failed to load the Windows UI font");
        }
#endif
    }

private:
    bool m_contextCreated = false;
    bool m_glfwBackendInitialized = false;
    bool m_openglBackendInitialized = false;
};
} // namespace

struct Application::State {
    State(Application* owner, int width, int height) : glfw(Application::glfwErrorCallback) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_CONTEXT_DEBUG, GLFW_TRUE);

        window.reset(glfwCreateWindow(width, height, "Venusaur", nullptr, nullptr));
        if (!window) {
            throw std::runtime_error("Failed to create a GLFW window");
        }

        glfwSetKeyCallback(window.get(), Application::glfwKeyCallback);
        glfwSetWindowUserPointer(window.get(), owner);
        glfwSetWindowSizeLimits(window.get(), width, height, GLFW_DONT_CARE, GLFW_DONT_CARE);
        glfwSetWindowAspectRatio(window.get(), width, height);
        glfwSetWindowSizeCallback(window.get(), Application::glfwResizeCallback);
        glfwMakeContextCurrent(window.get());
        glfwSwapInterval(1);

        auto output = RenderTarget::create(width, height);
        if (!output) {
            throw std::runtime_error(describe(output.error()));
        }
        outputBuffer = std::move(*output);

        auto presenter = Rasterizer::create();
        if (!presenter) {
            throw std::runtime_error(describe(presenter.error()));
        }
        rasterizer = std::move(*presenter);
        imgui.initialize(window.get());
    }

    GlfwRuntime glfw;
    Window window;
    std::shared_ptr<RayTracer> renderer;
    std::shared_ptr<RenderTarget> outputBuffer;
    std::shared_ptr<Rasterizer> rasterizer;
    ImGuiSession imgui;
};

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

Application::Application(int width, int height)
    : m_state(std::make_unique<State>(this, width, height)), m_width(width), m_height(height) {}

Application::~Application() = default;

std::shared_ptr<RenderTarget> Application::GetOutputBuffer() const {
    return m_state->outputBuffer;
}

void Application::SetRenderer(std::shared_ptr<RayTracer> renderer) {
    if (!renderer) {
        throw std::invalid_argument("Application renderer must not be null");
    }
    m_state->renderer = std::move(renderer);
}

void Application::run() {
    if (!m_state->renderer) {
        throw std::logic_error("Application renderer is not configured");
    }

    while (!glfwWindowShouldClose(m_state->window.get())) {
        auto startPoint = std::chrono::high_resolution_clock::now();

        if (auto result = m_state->renderer->render(m_state->outputBuffer); !result) {
            throw std::runtime_error(describe(result.error()));
        }
        m_state->rasterizer->render(m_width, m_height);

        auto endPoint = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endPoint - startPoint);

        glfwSetWindowTitle(m_state->window.get(), std::format("Venusaur - {}ms", duration.count()).c_str());

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

        glfwSwapBuffers(m_state->window.get());
        glfwPollEvents();
    }
}
} // namespace venusaur
