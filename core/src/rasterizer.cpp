#include <venusaur/rasterizer.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <spdlog/spdlog.h>

namespace venusaur {
namespace {
Result<GlShader> createGLShader(std::string_view source, GLuint shader_type) {
    GlShader shader{glCreateShader(shader_type)};
    const GLchar* source_data = reinterpret_cast<const GLchar*>(source.data());
    glShaderSource(shader.get(), 1, &source_data, nullptr);
    glCompileShader(shader.get());

    GLint is_compiled = 0;
    glGetShaderiv(shader.get(), GL_COMPILE_STATUS, &is_compiled);
    if (is_compiled == GL_FALSE) {
        GLint log_length = 0;
        glGetShaderiv(shader.get(), GL_INFO_LOG_LENGTH, &log_length);

        std::string info_log(log_length, '\0');
        glGetShaderInfoLog(shader.get(), log_length, nullptr, info_log.data());
        return std::unexpected(Error{
            .domain = ErrorDomain::opengl,
            .operation = "glCompileShader",
            .message = std::move(info_log),
        });
    }

    return shader;
}

Result<GlProgram> createGLProgram(std::string_view vert_src, std::string_view frag_src) {
    auto vertShader = createGLShader(vert_src, GL_VERTEX_SHADER);
    if (!vertShader) {
        return std::unexpected(std::move(vertShader.error()));
    }

    auto fragShader = createGLShader(frag_src, GL_FRAGMENT_SHADER);
    if (!fragShader) {
        return std::unexpected(std::move(fragShader.error()));
    }

    GlProgram program{glCreateProgram()};
    glAttachShader(program.get(), vertShader->get());
    glAttachShader(program.get(), fragShader->get());
    glLinkProgram(program.get());

    GLint is_linked = 0;
    glGetProgramiv(program.get(), GL_LINK_STATUS, &is_linked);
    if (is_linked == GL_FALSE) {
        GLint log_length = 0;
        glGetProgramiv(program.get(), GL_INFO_LOG_LENGTH, &log_length);

        std::string info_log(log_length, '\0');
        glGetProgramInfoLog(program.get(), log_length, nullptr, info_log.data());
        return std::unexpected(Error{
            .domain = ErrorDomain::opengl,
            .operation = "glLinkProgram",
            .message = std::move(info_log),
        });
    }

    glDetachShader(program.get(), vertShader->get());
    glDetachShader(program.get(), fragShader->get());

    return program;
}

constexpr std::string_view kVertexSource = R"(
    #version 460 core

    out vec2 texcoord;

    void main()
    {
        texcoord = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(texcoord * 2 - 1, 0, 1);
    }
)";

constexpr std::string_view kFragmentSource = R"(
    #version 460 core

    layout(binding = 0) uniform sampler2D tex;
    in vec2 texcoord;
    out vec4 color;
    
    void main()
    {
        vec2 uv = vec2(texcoord.x, 1.0 - texcoord.y);
        color = texture(tex, uv);
    }
)";

constexpr std::string_view glSourceToString(GLenum source) noexcept {
    switch (source) {
    case GL_DEBUG_SOURCE_API:
        return "API";
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        return "Window System";
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        return "Shader Compiler";
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        return "Third Party";
    case GL_DEBUG_SOURCE_APPLICATION:
        return "Application";
    case GL_DEBUG_SOURCE_OTHER:
        return "Other";
    default:
        return "Unknown";
    }
}

constexpr std::string_view glTypeToString(GLenum type) noexcept {
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        return "Error";
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        return "Deprecated Behavior";
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        return "Undefined Behavior";
    case GL_DEBUG_TYPE_PORTABILITY:
        return "Portability";
    case GL_DEBUG_TYPE_PERFORMANCE:
        return "Performance";
    case GL_DEBUG_TYPE_MARKER:
        return "Marker";
    case GL_DEBUG_TYPE_PUSH_GROUP:
        return "Push Group";
    case GL_DEBUG_TYPE_POP_GROUP:
        return "Pop Group";
    case GL_DEBUG_TYPE_OTHER:
        return "Other";
    default:
        return "Unknown";
    }
}

constexpr spdlog::level::level_enum glSeverityToSpdlogLevel(GLenum severity) noexcept {
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        return spdlog::level::critical;
    case GL_DEBUG_SEVERITY_MEDIUM:
        return spdlog::level::err;
    case GL_DEBUG_SEVERITY_LOW:
        return spdlog::level::warn;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        return spdlog::level::info;
    default:
        return spdlog::level::debug;
    }
}

void APIENTRY messageCallback(GLenum source,
                              GLenum type,
                              GLuint id,
                              GLenum severity,
                              GLsizei length,
                              GLchar const* message,
                              void const* user_param) {
    // ignore non-significant error/warning codes
    if (id == 131154 || /*id == 131169 ||*/ id == 131185 /*|| id == 131218*/ || id == 131204) [[unlikely]] {
        return;
    }

    auto logLevel = glSeverityToSpdlogLevel(severity);
    if (spdlog::should_log(logLevel)) {
        spdlog::log(logLevel, "[OpenGL][{}][{}][{}] {}", glSourceToString(source), glTypeToString(type), id, message);
    }
}
} // anonymous namespace

Result<std::shared_ptr<Rasterizer>> Rasterizer::create() {
    if (gl3wInit()) {
        return std::unexpected(Error{
            .domain = ErrorDomain::opengl,
            .operation = "gl3wInit",
            .message = "Failed to initialize OpenGL function loading",
        });
    }

    auto rasterizer = std::shared_ptr<Rasterizer>(new Rasterizer{});

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(messageCallback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

    GLint colorSpace = 0;
    glGetNamedFramebufferAttachmentParameteriv(0, GL_FRONT_LEFT, GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, &colorSpace);
    switch (colorSpace) {
    case GL_RGB:
        glEnable(GL_FRAMEBUFFER_SRGB);
        break;
    case GL_LINEAR:
        glDisable(GL_FRAMEBUFFER_SRGB);
        break;
    }

    auto program = createGLProgram(kVertexSource, kFragmentSource);
    if (!program) {
        return std::unexpected(std::move(program.error()));
    }
    rasterizer->m_program = std::move(*program);
    glUseProgram(rasterizer->m_program.get());

    GLuint vertexArray = 0;
    glCreateVertexArrays(1, &vertexArray);
    rasterizer->m_vertexArray.reset(vertexArray);
    if (auto result = checkGl("create rasterizer state"); !result) {
        return std::unexpected(std::move(result.error()));
    }

    return rasterizer;
}

void Rasterizer::render(GLuint width, GLuint height) {
    glViewport(0, 0, width, height);
    glScissor(0, 0, width, height);
    constexpr GLfloat clearColor[] = {0.0f, 0.0f, 0.0f, 1.0f};
    constexpr GLfloat clearDepth = 0.0f;
    glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor);
    glClearNamedFramebufferfv(0, GL_DEPTH, 0, &clearDepth);

    glBindVertexArray(m_vertexArray.get());
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
} // namespace venusaur
