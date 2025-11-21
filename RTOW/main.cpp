#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include <nvrtc.h>

#include <application.h>

#include "exception.h"
#include "renderer_image.h"
#include "renderer_background.h"
#include "renderer_sphere.h"
#include "renderer_normal.h"
#include "renderer_antialiasing.h"
#include "renderer_diffuse.h"

int main(int argc, char* argv[]) 
{
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    auto app = std::make_unique<Venusaur::Application>(image_width, image_height);

    auto file = std::ifstream{"cuda/diffuse.cu"};
    auto source = std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
    file.close();

    auto program = nvrtcProgram{};
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&program, source.c_str(), "diffuse.cu", 0, NULL, NULL));
    const char* const options[] = {
        "-std=c++20",
        "-optix-ir",
        "-IC:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.0.0\\include",
        "-Icuda",
        "--use_fast_math",
    };
    NVRTC_SAFE_CALL(nvrtcCompileProgram(program, 5, options));
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

    app->SetRenderer(std::make_shared<RayTracingInOneWeekend::RendererDiffuse>(app->GetOutputBuffer(), optixir));
    
    while (app->IsRunning()) 
    {
        app->Update();
    }

    return 0;
}