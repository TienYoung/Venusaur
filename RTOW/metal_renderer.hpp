#pragma once

#include <cstddef>

#include <array>
#include <memory>

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <glm/glm.hpp>

#include <venusaur/exception.hpp>

#include <venusaur/ray_tracer.hpp>
#include <venusaur/render_target.hpp>

#include "cuda/metal.h"

namespace rtow {
class MetalRenderer : venusaur::IRenderable {
private:
    union ProgramGroup {
        struct {
            OptixProgramGroup raygen;
            OptixProgramGroup miss;
            OptixProgramGroup hitgroup_lambertian;
            OptixProgramGroup hitgroup_metal;
        };
        std::array<OptixProgramGroup, 4> array;
    };

public:
    MetalRenderer(std::shared_ptr<venusaur::RayTracer> ray_tracer, const std::vector<char>& optixIR) {
        ray_tracer->setRenderCallback(
            [this](uchar4* image, uint32_t width, uint32_t height) { return this->setupParams(image, width, height); });

        auto accelBuildOption = OptixAccelBuildOptions{.buildFlags = OPTIX_BUILD_FLAG_NONE,
                                                       .operation = OPTIX_BUILD_OPERATION_BUILD,
                                                       .motionOptions = {
                                                           .numKeys = 0,
                                                           .flags = OPTIX_MOTION_FLAG_NONE,
                                                           .timeBegin = 0.f,
                                                           .timeEnd = 0.f,
                                                       }};

        std::array sphereCenter = {
            make_float3(0.0f, -100.5f, -1.0f),
            make_float3(0.0f, 0.0f, -1.2f),
            make_float3(-1.0f, 0.0f, -1.0f),
            make_float3(1.0f, 0.0f, -1.0f),
        };

        std::array sphereRadius = {
            100.0f,
            0.5f,
            0.5f,
            0.5f,
        };

        std::array indices = {
            0,
            1,
            2,
            3,
        };

        CUdeviceptr d_center_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_center_buffer), sizeof(float3) * sphereCenter.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_center_buffer),
                              sphereCenter.data(),
                              sizeof(sphereCenter),
                              cudaMemcpyHostToDevice));

        CUdeviceptr d_radius_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius_buffer), sizeof(float) * sphereRadius.size()));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_radius_buffer),
                              sphereRadius.data(),
                              sizeof(sphereRadius),
                              cudaMemcpyHostToDevice));

        CUdeviceptr d_sbt_index_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_sbt_index_buffer), sizeof(indices)));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_sbt_index_buffer), indices.data(), sizeof(indices), cudaMemcpyHostToDevice));

        unsigned int sphereInputFlags[] = {
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        };
        OptixBuildInput sphereInput = {.type = OPTIX_BUILD_INPUT_TYPE_SPHERES,
                                       .sphereArray = {
                                           .vertexBuffers = &d_center_buffer,
                                           .vertexStrideInBytes = sizeof(float3),
                                           .numVertices = sphereCenter.size(),
                                           .radiusBuffers = &d_radius_buffer,
                                           .radiusStrideInBytes = sizeof(float),
                                           .singleRadius = false,
                                           .flags = sphereInputFlags,
                                           .numSbtRecords = indices.size(),
                                           .sbtIndexOffsetBuffer = d_sbt_index_buffer,
                                           .sbtIndexOffsetSizeInBytes = sizeof(int),
                                           .sbtIndexOffsetStrideInBytes = 0,
                                           .primitiveIndexOffset = 0,
                                       }};

        m_gasHandle = ray_tracer->createAccelBuffer(accelBuildOption, sphereInput);

        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_center_buffer)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_radius_buffer)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_sbt_index_buffer)));

        OptixModuleCompileOptions moduleCompileOptions = {
            .maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
#ifdef _DEBUG
            .optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0,
            .debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL,
#endif
            .boundValues = nullptr,
            .numBoundValues = 0,
            .numPayloadTypes = 0,
            .payloadTypes = nullptr,
        };

        OptixPipelineCompileOptions pipelineCompileOptions = {
            .usesMotionBlur = false,
            .traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
            .numPayloadValues = sizeof(MetalPayload) / sizeof(unsigned int),
            .numAttributeValues = 0,
            .exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE,
            .pipelineLaunchParamsVariableName = "params",
            .usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE,
            .allowOpacityMicromaps = false,
            .allowClusteredGeometry = false,
        };

        OptixModule module = nullptr;

        char log[2048];
        size_t log_length = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreate(ray_tracer->getOptixContext(),
                                          &moduleCompileOptions,
                                          &pipelineCompileOptions,
                                          optixIR.data(),
                                          optixIR.size(),
                                          log,
                                          &log_length,
                                          &module));

        OptixModule sphereModuleIS = nullptr;
        OptixBuiltinISOptions sphereISOptions = {
            .builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE,
            .usesMotionBlur = false,
            .buildFlags = OPTIX_BUILD_FLAG_NONE,
            .curveEndcapFlags = OPTIX_CURVE_ENDCAP_DEFAULT,
        };
        OPTIX_CHECK_LOG(optixBuiltinISModuleGet(ray_tracer->getOptixContext(),
                                                &moduleCompileOptions,
                                                &pipelineCompileOptions,
                                                &sphereISOptions,
                                                &sphereModuleIS));

        std::array programGroupDesc = {
            OptixProgramGroupDesc{.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN,
                                  .flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
                                  .raygen =
                                      {
                                          .module = module,
                                          .entryFunctionName = "__raygen__",
                                      }},
            OptixProgramGroupDesc{.kind = OPTIX_PROGRAM_GROUP_KIND_MISS,
                                  .flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
                                  .miss =
                                      {
                                          .module = module,
                                          .entryFunctionName = "__miss__",
                                      }},
            OptixProgramGroupDesc{.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                                  .flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
                                  .hitgroup =
                                      {
                                          .moduleCH = module,
                                          .entryFunctionNameCH = "__closesthit__lambertian",
                                          .moduleAH = nullptr,
                                          .entryFunctionNameAH = nullptr,
                                          .moduleIS = sphereModuleIS,
                                          .entryFunctionNameIS = nullptr,
                                      }},
            OptixProgramGroupDesc{.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP,
                                  .flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
                                  .hitgroup =
                                      {
                                          .moduleCH = module,
                                          .entryFunctionNameCH = "__closesthit__metal",
                                          .moduleAH = nullptr,
                                          .entryFunctionNameAH = nullptr,
                                          .moduleIS = sphereModuleIS,
                                          .entryFunctionNameIS = nullptr,
                                      }},
        };

        ProgramGroup programGroup = {};

        OptixProgramGroupOptions programGroupOptions = {
            .payloadType = nullptr,
        };
        {
            char log[2048];
            size_t log_length = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(ray_tracer->getOptixContext(),
                                                    programGroupDesc.data(),
                                                    programGroupDesc.size(),
                                                    &programGroupOptions,
                                                    log,
                                                    &log_length,
                                                    programGroup.array.data()));
        }

        OptixPipelineLinkOptions pipelineLinkOptions = {
            .maxTraceDepth = 1,
            .maxContinuationCallableDepth = 0,
            .maxDirectCallableDepthFromState = 0,
            .maxDirectCallableDepthFromTraversal = 0,
            .maxTraversableGraphDepth = 0,
        };

        ray_tracer->setupPipeline(pipelineCompileOptions, pipelineLinkOptions, programGroup.array);

        CUdeviceptr d_raygenRecord = NULL;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygenRecord), sizeof(RayGenSbtRecord)));
        RayGenSbtRecord raygenRecord = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(programGroup.raygen, &raygenRecord));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_raygenRecord), &raygenRecord, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixProgramGroupDestroy(programGroup.raygen));

        CUdeviceptr d_missRecordBase = NULL;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_missRecordBase), sizeof(MissSbtRecord)));
        MissSbtRecord missRecord = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(programGroup.miss, &missRecord));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void*>(d_missRecordBase), &missRecord, sizeof(MissSbtRecord), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixProgramGroupDestroy(programGroup.miss));

        std::array hitGroupRecords = {
            HitGroupSbtRecord{
                .data =
                    {
                        .lambertian = {.albedo = float3{.x = 0.8f, .y = 0.8f, .z = 0.0f}},
                    },
            },
            HitGroupSbtRecord{
                .data =
                    {
                        .lambertian = {.albedo = float3{.x = 0.1f, .y = 0.2f, .z = 0.5f}},
                    },
            },
            HitGroupSbtRecord{
                .data =
                    {
                        .metal =
                            {
                                .albedo = float3{.x = 0.8f, .y = 0.8f, .z = 0.8f},
                                .fuzz = 0.3,
                            },
                    },
            },
            HitGroupSbtRecord{
                .data =
                    {
                        .metal =
                            {
                                .albedo = float3{.x = 0.8f, .y = 0.6f, .z = 0.2f},
                                .fuzz = 1.0f,
                            },
                    },
            },
        };

        OPTIX_CHECK(optixSbtRecordPackHeader(programGroup.hitgroup_lambertian, hitGroupRecords[0].header));
        OPTIX_CHECK(optixSbtRecordPackHeader(programGroup.hitgroup_lambertian, hitGroupRecords[1].header));
        OPTIX_CHECK(optixSbtRecordPackHeader(programGroup.hitgroup_metal, hitGroupRecords[2].header));
        OPTIX_CHECK(optixSbtRecordPackHeader(programGroup.hitgroup_metal, hitGroupRecords[3].header));

        OPTIX_CHECK(optixProgramGroupDestroy(programGroup.hitgroup_lambertian));
        OPTIX_CHECK(optixProgramGroupDestroy(programGroup.hitgroup_metal));

        CUdeviceptr d_hitGroupRecordBase = NULL;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitGroupRecordBase), sizeof(hitGroupRecords)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitGroupRecordBase),
                              hitGroupRecords.data(),
                              sizeof(hitGroupRecords),
                              cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixModuleDestroy(module));

        // char location[2048] = "";
        // OPTIX_CHECK(optixDeviceContextGetCacheLocation(m_context, location, 2048));
        // std::cout << "location:" << location << std::endl;

        ray_tracer->setupShaderBindingTable({
            .raygenRecord = d_raygenRecord,
            .missRecordBase = d_missRecordBase,
            .missRecordStrideInBytes = sizeof(MissSbtRecord),
            .missRecordCount = 1,
            .hitgroupRecordBase = d_hitGroupRecordBase,
            .hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord),
            .hitgroupRecordCount = hitGroupRecords.size(),
            .callablesRecordBase = NULL,
            .callablesRecordStrideInBytes = 0,
            .callablesRecordCount = 0,
        });

        ray_tracer->mallocParamsOnDevice(sizeof(MetalParams));
    }

    ~MetalRenderer() {}

private:
    typedef venusaur::RayTracer::SbtRecord<void> RayGenSbtRecord;
    typedef venusaur::RayTracer::SbtRecord<void> MissSbtRecord;
    typedef venusaur::RayTracer::SbtRecord<Material> HitGroupSbtRecord;

    OptixTraversableHandle m_gasHandle = 0;

    unsigned int m_subframeIndex = 0;

    MetalParams m_params;

    std::span<const std::byte> setupParams(uchar4* image, uint32_t width, uint32_t height) {
        int image_width = width;
        int image_height = height;

        // Camera
        auto focal_length = 1.0;
        auto viewport_height = 2.0;
        auto viewport_width = viewport_height * (double(image_width) / image_height);
        auto camera_center = glm::vec3(0, 0, 0);
        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        auto viewport_u = glm::vec3(viewport_width, 0, 0);
        auto viewport_v = glm::vec3(0, -viewport_height, 0);
        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        auto pixel_delta_u = viewport_u / (float)image_width;
        auto pixel_delta_v = viewport_v / (float)image_height;
        // Calculate the location of the upper left pixel.
        auto viewport_upper_left =
            camera_center - glm::vec3(0, 0, focal_length) - viewport_u / 2.0f - viewport_v / 2.0f;
        auto pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        m_params = {
            .image = image,
            .camera_center = make_float3(camera_center.x, camera_center.y, camera_center.z),
            .pixel00_loc = make_float3(pixel00_loc.x, pixel00_loc.y, pixel00_loc.z),
            .pixel_delta_u = make_float3(pixel_delta_u.x, pixel_delta_u.y, pixel_delta_u.z),
            .pixel_delta_v = make_float3(pixel_delta_v.x, pixel_delta_v.y, pixel_delta_v.z),
            .samples_per_pixel = 100,
            .subframe_index = m_subframeIndex++,
            .handle = m_gasHandle,
        };

        return std::span<const std::byte>{reinterpret_cast<std::byte*>(&m_params), sizeof(MetalParams)};
    }
};
} // namespace rtow