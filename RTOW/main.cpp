#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nvrtc.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>

#include <venusaur/application.hpp>

#include "metal_renderer.hpp"

namespace {
void logCritical(const venusaur::Error& error) {
    spdlog::critical("[{}:{}] {}: {}",
                     venusaur::toString(error.domain),
                     error.code,
                     error.operation,
                     error.message);
}

int run() {
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    auto app = std::make_unique<venusaur::Application>(image_width, image_height);

    auto filePath = std::filesystem::path{"cuda/metal.cu"};
    auto file = std::ifstream{filePath};
    auto source = std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    file.close();

    auto program = nvrtcProgram{};
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program, source.c_str(), "metal.cu", 0, nullptr, nullptr));
#ifdef _WIN32
    char* cuda_path = nullptr;
    size_t len;
    auto err = _dupenv_s(&cuda_path, &len, "CUDA_PATH");
    if (err) {
        exit(EXIT_FAILURE);
    }
    char* optix_install_dir = nullptr;
    err = _dupenv_s(&optix_install_dir, &len, "OPTIX_INSTALL_DIR");
    if (err) {
        exit(EXIT_FAILURE);
    }
#else
    auto cuda_path = std::getenv("CUDA_PATH");
    auto optix_install_dir = std::getenv("OPTIX_INSTALL_DIR");
#endif
    auto cudaInclude = fmt::format("-I{}/include", cuda_path);
    auto ccclInclude = fmt::format("-I{}/include/cccl", cuda_path);
    auto optixInclude = fmt::format("-I{}/include", optix_install_dir);

    std::array options = {
        "-std=c++20",
        "-optix-ir",
#ifdef _DEBUG
        "-G",
#endif
        cudaInclude.c_str(),
        ccclInclude.c_str(),
        optixInclude.c_str(),
        "-Icuda",
        "--use_fast_math",
    };
    NVRTC_SAFE_CALL(nvrtcCompileProgram(program, options.size(), options.data()));
    auto size = size_t{};
    auto log = std::string{};
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(program, &size));
    log.resize(size);
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(program, log.data()));
    if (!log.empty() && log.back() == '\0') {
        log.pop_back();
    }
    if (!log.empty()) {
        spdlog::info("NVRTC compile log:\n{}", log);
    }
    auto optixir = std::vector<char>{};
    NVRTC_SAFE_CALL(nvrtcGetOptiXIRSize(program, &size));
    optixir.resize(size);
    NVRTC_SAFE_CALL(nvrtcGetOptiXIR(program, optixir.data()));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

    auto rayTracerResult = venusaur::RayTracer::create();
    if (!rayTracerResult) {
        logCritical(rayTracerResult.error());
        return EXIT_FAILURE;
    }
    auto ray_tracer = std::move(*rayTracerResult);
    app->SetRenderer(ray_tracer);

    auto rendererResult = rtow::MetalRenderer::create(ray_tracer, optixir);
    if (!rendererResult) {
        logCritical(rendererResult.error());
        return EXIT_FAILURE;
    }
    auto renderer = std::move(*rendererResult);

    app->run();

    return 0;
}
} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        spdlog::critical("Venusaur failed: {}", error.what());
        return EXIT_FAILURE;
    }
}
