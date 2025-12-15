#include <array>
#include <cstddef>
#include <cstdlib>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <filesystem>

#include <nvrtc.h>

#include <application.h>

#include "renderer_metal.h"

int main(int argc, char* argv[]) 
{
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    auto app = std::make_unique<Venusaur::Application>(image_width, image_height);

    auto filePath = std::filesystem::path{"cuda/metal.cu"};
    auto file = std::ifstream{filePath};
    auto source = std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    file.close();

    auto program = nvrtcProgram{};
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program, source.c_str(), "metal.cu", 0, nullptr, nullptr));
#ifdef WIN32
    char* cuda_path = nullptr;
    size_t len;
    auto err = _dupenv_s(&cuda_path, &len, "CUDA_PATH");
    if (err)
        exit(EXIT_FAILURE);
    char* optix_install_dir = nullptr;
    err = _dupenv_s(&optix_install_dir, &len, "OPTIX_INSTALL_DIR");
    if (err)
        exit(EXIT_FAILURE);
#else
    auto cuda_path = std::getenv("CUDA_PATH");
    auto optix_install_dir = std::getenv("OPTIX_INSTALL_DIR");
#endif
    auto cudaInclude = std::format("-I{}/include", cuda_path);
    auto optixInclude = std::format("-I{}/include", optix_install_dir);

    std::array options = {
        "-std=c++20",
        "-optix-ir",
        cudaInclude.c_str(),
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
    std::cerr << log.c_str() << std::endl;
    auto optixir= std::vector<char>{};
    NVRTC_SAFE_CALL(nvrtcGetOptiXIRSize(program, &size));
    optixir.resize(size);
    NVRTC_SAFE_CALL(nvrtcGetOptiXIR(program, optixir.data()));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

    app->SetRenderer(std::make_shared<RayTracingInOneWeekend::RendererMetal>(app->GetOutputBuffer(), optixir));
    
    while (app->IsRunning()) 
    {
        auto optix7cache = std::filesystem::path("C:\\Users\\const\\AppData\\Local\\NVIDIA\\OptixCache\\optix7cache.db");
        auto cacheTime = std::filesystem::last_write_time(optix7cache);
        auto sourceTime = std::filesystem::last_write_time(filePath);
        // if(false)
        // {
        //     auto filePath = std::filesystem::path{"cuda/diffuse.cu"};
        //     auto file = std::ifstream{filePath};
        //     auto source = std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
        //     file.close();

        //     auto program = nvrtcProgram{};
        //     NVRTC_SAFE_CALL(nvrtcCreateProgram(&program, source.c_str(), "diffuse.cu", 0, NULL, NULL));
        //     const char* const options[] = {
        //         "-std=c++20",
        //         "-optix-ir",
        //         "-IC:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.1.0\\include",
        //         "-Icuda",
        //         "--use_fast_math",
        //     };
        //     NVRTC_SAFE_CALL(nvrtcCompileProgram(program, 5, options));
        //     auto size = size_t{};
        //     auto log = std::string{};
        //     NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(program, &size));
        //     log.resize(size);
        //     NVRTC_SAFE_CALL(nvrtcGetProgramLog(program, log.data()));
        //     std::cerr << log.c_str() << std::endl;
        //     auto optixir= std::vector<char>{};
        //     NVRTC_SAFE_CALL(nvrtcGetOptiXIRSize(program, &size));
        //     optixir.resize(size);
        //     NVRTC_SAFE_CALL(nvrtcGetOptiXIR(program, optixir.data()));
        //     NVRTC_SAFE_CALL(nvrtcDestroyProgram(&program));

        //     app->SetRenderer(std::make_shared<RayTracingInOneWeekend::RendererDiffuse>(app->GetOutputBuffer(), optixir));
        // }

        app->Update();  
    }

    return 0;
}