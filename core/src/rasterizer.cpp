#include <venusuar/rasterizer.hpp>

#include <spdlog/spdlog.h>
#include <string_view>

namespace venusaur {
namespace {
GLuint createGLShader(std::string_view source, GLuint shader_type) {
    GLuint shader = glCreateShader(shader_type);
    const GLchar* source_data = reinterpret_cast<const GLchar*>(source.data());
    glShaderSource(shader, 1, &source_data, nullptr);
    glCompileShader(shader);

    GLint is_compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
    if (is_compiled == GL_FALSE) {
        GLint log_length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

        std::string info_log(log_length, '\0');
        glGetShaderInfoLog(shader, log_length, nullptr, info_log.data());

        spdlog::error("[Shader Compile Error] {}", info_log);
        glDeleteShader(shader);

        return 0;
    }

    return shader;
}

GLuint createGLProgram(std::string_view vert_src, std::string_view frag_src) {
    GLuint vert_shader = createGLShader(vert_src, GL_VERTEX_SHADER);
    if (vert_shader == 0) {
        return 0;
    }

    GLuint frag_shader = createGLShader(frag_src, GL_FRAGMENT_SHADER);
    if (frag_shader == 0) {
        glDeleteShader(vert_shader);
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert_shader);
    glAttachShader(program, frag_shader);
    glLinkProgram(program);

    GLint is_linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
    if (is_linked == GL_FALSE) {
        GLint log_length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);

        std::string info_log(log_length, '\0');
        glGetProgramInfoLog(program, log_length, nullptr, info_log.data());

        spdlog::error("[Program Link Error] {}", info_log);
        glDeleteProgram(program);
        glDeleteShader(vert_shader);
        glDeleteShader(frag_shader);

        return 0;
    }

    glDetachShader(program, vert_shader);
    glDetachShader(program, frag_shader);

    return program;
}

GLint getGLUniformLocation(GLuint program, std::string_view name) {
    GLint loc = glGetUniformLocation(program, name.data());
    if (loc == -1) {
        throw std::runtime_error(std::format("Failed to get uniform loc for '{}'", name).c_str());
    }
    return loc;
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

Rasterizer::Rasterizer() {
    if (gl3wInit()) {
        throw std::runtime_error("Failed to initialize GL");
    }

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

    // Create program
    m_program = createGLProgram(kVertexSource, kFragmentSource);
    glUseProgram(m_program);

    // Create VAO
    glCreateVertexArrays(1, &m_vao);
}

Rasterizer::~Rasterizer() {
    glDeleteProgram(m_program);
    glDeleteVertexArrays(1, &m_vao);
}

void Rasterizer::render(GLuint width, GLuint height) {
    glViewport(0, 0, width, height);
    glScissor(0, 0, width, height);
    constexpr GLfloat clearColor[] = {0.0f, 0.0f, 0.0f, 1.0f};
    constexpr GLfloat clearDepth = 0.0f;
    glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor);
    glClearNamedFramebufferfv(0, GL_DEPTH, 0, &clearDepth);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}
} // namespace venusaur