#include "GL/gl3w.h"
#include "GL/glcorearb.h"
#include "rasterizer.h"

#include <string>

#include <spdlog/spdlog.h>
#include <string_view>

namespace Venusaur
{
     namespace
    {
        GLuint createGLShader(std::string_view source, GLuint shader_type)
        {
            GLuint shader = glCreateShader(shader_type);
            {
                const GLchar* source_data = reinterpret_cast<const GLchar*>(source.data());
                glShaderSource(shader, 1, &source_data, nullptr);
                glCompileShader(shader);

                GLint is_compiled = 0;
                glGetShaderiv(shader, GL_COMPILE_STATUS, &is_compiled);
                if (is_compiled == GL_FALSE)
                {
                    GLint log_length = 0;
                    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

                    std::string info_log(log_length, '\0');
                    glGetShaderInfoLog(shader, log_length, nullptr, info_log.data());
                    
                    spdlog::error("[Shader Compile Error] {}", info_log);
                    glDeleteShader(shader);

                    return 0;
                }
            }

            return shader;
        }


        GLuint createGLProgram(std::string_view vert_src, std::string_view frag_src)
        {
            GLuint vert_shader = createGLShader(vert_src, GL_VERTEX_SHADER);
            if (vert_shader == 0)
                return 0;

            GLuint frag_shader = createGLShader(frag_src, GL_FRAGMENT_SHADER);
            if (frag_shader == 0)
            {
                glDeleteShader(vert_shader);
                return 0;
            }

            GLuint program = glCreateProgram();
            glAttachShader(program, vert_shader);
            glAttachShader(program, frag_shader);
            glLinkProgram(program);

            GLint is_linked = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &is_linked);
            if (is_linked == GL_FALSE)
            {
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


        GLint getGLUniformLocation(GLuint program, std::string_view name)
        {
            GLint loc = glGetUniformLocation(program, name.data());
            if (loc == -1) {
                throw std::runtime_error(std::format("Failed to get uniform loc for '{}'", name).c_str());
            }
            return loc;
        }

    } // anonymous namespace

    constexpr std::string_view vertex_source = R"(
        #version 460 core

        out vec2 texcoord;

        void main()
        {
            texcoord = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            gl_Position = vec4(texcoord * 2 - 1, 0, 1);
        }
    )";

    constexpr std::string_view fragment_source = R"(
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

    

    void APIENTRY messageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, GLchar const* message, void const* user_param)
    {
        // ignore non-significant error/warning codes
        if (id == 131154 || /*id == 131169 ||*/ id == 131185 /*|| id == 131218*/ || id == 131204)
            return;

        std::string sourceStr, typeStr, severityStr;

        switch (source)	{
            case GL_DEBUG_SOURCE_API:             sourceStr = "API"; break;
            case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   sourceStr = "Window System"; break;
            case GL_DEBUG_SOURCE_SHADER_COMPILER: sourceStr = "Shader Compiler"; break;
            case GL_DEBUG_SOURCE_THIRD_PARTY:     sourceStr = "Third Party"; break;
            case GL_DEBUG_SOURCE_APPLICATION:     sourceStr = "Application"; break;
            case GL_DEBUG_SOURCE_OTHER:           sourceStr = "Other"; break;
            default:							  sourceStr = "Unknown"; break;
        } 

        switch (type) {
            case GL_DEBUG_TYPE_ERROR:               typeStr = "Error"; break;
            case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typeStr = "Deprecated Behavior"; break;
            case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typeStr = "Undefined Behavior"; break;
            case GL_DEBUG_TYPE_PORTABILITY:         typeStr = "Portability"; break;
            case GL_DEBUG_TYPE_PERFORMANCE:         typeStr = "Performance"; break;
            case GL_DEBUG_TYPE_MARKER:              typeStr = "Marker"; break;
            case GL_DEBUG_TYPE_PUSH_GROUP:          typeStr = "Push Group"; break;
            case GL_DEBUG_TYPE_POP_GROUP:           typeStr = "Pop Group"; break;
            case GL_DEBUG_TYPE_OTHER:               typeStr = "Other"; break;
            default:								typeStr = "Unknown"; break;
        }

        switch (severity) {
            case GL_DEBUG_SEVERITY_HIGH: 
                spdlog::critical("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
                break;
            case GL_DEBUG_SEVERITY_MEDIUM:
                spdlog::error("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
                break;
            case GL_DEBUG_SEVERITY_LOW:
                spdlog::warn("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
                break;
            case GL_DEBUG_SEVERITY_NOTIFICATION:
                spdlog::info("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
                break;
            default:
                spdlog::debug("[OpenGL][{}][{}][{}] {}", sourceStr, typeStr, id, message);
                break;
        }
    }

    Rasterizer::Rasterizer()
    {
        if (gl3wInit())
        {
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
        switch (colorSpace) 
        {
        case GL_RGB:
            glEnable(GL_FRAMEBUFFER_SRGB);
            break;
        case GL_LINEAR:
            glDisable(GL_FRAMEBUFFER_SRGB);
            break;
        }

        // Create program
        m_program = createGLProgram(vertex_source, fragment_source);
        glUseProgram(m_program);

        // Create VAO
        glCreateVertexArrays(1, &m_vao);
    }

    Rasterizer::~Rasterizer()
    {
        glDeleteProgram(m_program);
        glDeleteVertexArrays(1, &m_vao);
    }

    void Rasterizer::Render(GLuint width, GLuint height)
    {
        glViewport(0, 0, width, height);
        glScissor(0, 0, width, height);
        constexpr GLfloat clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
        constexpr GLfloat clearDepth = 0.0f;
        glClearNamedFramebufferfv(0, GL_COLOR, 0, clearColor);
        glClearNamedFramebufferfv(0, GL_DEPTH, 0, &clearDepth);

        glBindVertexArray(m_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
}