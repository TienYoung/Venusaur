#pragma once

#include <expected>
#include <string>
#include <string_view>

#include <spdlog/fmt/fmt.h>

namespace venusaur {
enum class ErrorDomain {
    application,
    opengl,
    cuda,
    optix,
    nvrtc,
    system,
};

struct Error {
    ErrorDomain domain = ErrorDomain::application;
    int code = 0;
    std::string operation;
    std::string message;
};

template <typename T> using Result = std::expected<T, Error>;

[[nodiscard]] constexpr std::string_view toString(ErrorDomain domain) noexcept {
    switch (domain) {
    case ErrorDomain::application:
        return "Application";
    case ErrorDomain::opengl:
        return "OpenGL";
    case ErrorDomain::cuda:
        return "CUDA";
    case ErrorDomain::optix:
        return "OptiX";
    case ErrorDomain::nvrtc:
        return "NVRTC";
    case ErrorDomain::system:
        return "System";
    }
    return "Unknown";
}

[[nodiscard]] inline std::string describe(const Error& error) {
    return fmt::format("[{}:{}] {}: {}", toString(error.domain), error.code, error.operation, error.message);
}
} // namespace venusaur
