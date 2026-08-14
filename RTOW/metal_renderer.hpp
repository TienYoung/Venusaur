#pragma once

#include <cstddef>

#include <array>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <optix_stubs.h>

#include <glm/glm.hpp>

#include <venusaur/gpu_resources.hpp>
#include <venusaur/ray_tracer.hpp>
#include <venusaur/render_target.hpp>
#include <venusaur/result.hpp>

#include "cuda/metal.h"

namespace rtow {
class MetalRenderer {
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
    [[nodiscard]] static venusaur::Result<std::shared_ptr<MetalRenderer>>
    create(std::shared_ptr<venusaur::RayTracer> ray_tracer, const std::vector<char>& optixIR) {
        auto renderer = std::shared_ptr<MetalRenderer>(new MetalRenderer{});
        if (auto result = renderer->initialize(ray_tracer, optixIR); !result) {
            return std::unexpected(std::move(result.error()));
        }

        std::weak_ptr<MetalRenderer> weakRenderer = renderer;
        ray_tracer->setRenderCallback(
            [weakRenderer](uchar4* image,
                           uint32_t width,
                           uint32_t height) -> venusaur::Result<std::span<const std::byte>> {
                auto locked = weakRenderer.lock();
                if (!locked) {
                    return std::unexpected(venusaur::Error{
                        .domain = venusaur::ErrorDomain::application,
                        .operation = "MetalRenderer::setupParams",
                        .message = "Metal renderer no longer exists",
                    });
                }
                return locked->setupParams(image, width, height);
            });

        return renderer;
    }

private:
    MetalRenderer() = default;

    [[nodiscard]] venusaur::Result<void> initialize(std::shared_ptr<venusaur::RayTracer> ray_tracer,
                                                    const std::vector<char>& optixIR) {

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

        auto centerBuffer = venusaur::allocateDeviceBuffer(sizeof(sphereCenter));
        if (!centerBuffer) {
            return std::unexpected(std::move(centerBuffer.error()));
        }
        if (auto result = venusaur::checkCuda(cudaMemcpy(reinterpret_cast<void*>(centerBuffer->get()),
                                                         sphereCenter.data(),
                                                         sizeof(sphereCenter),
                                                         cudaMemcpyHostToDevice),
                                               "copy sphere centers");
            !result) {
            return result;
        }

        auto radiusBuffer = venusaur::allocateDeviceBuffer(sizeof(sphereRadius));
        if (!radiusBuffer) {
            return std::unexpected(std::move(radiusBuffer.error()));
        }
        if (auto result = venusaur::checkCuda(cudaMemcpy(reinterpret_cast<void*>(radiusBuffer->get()),
                                                         sphereRadius.data(),
                                                         sizeof(sphereRadius),
                                                         cudaMemcpyHostToDevice),
                                               "copy sphere radii");
            !result) {
            return result;
        }

        auto sbtIndexBuffer = venusaur::allocateDeviceBuffer(sizeof(indices));
        if (!sbtIndexBuffer) {
            return std::unexpected(std::move(sbtIndexBuffer.error()));
        }
        if (auto result = venusaur::checkCuda(cudaMemcpy(reinterpret_cast<void*>(sbtIndexBuffer->get()),
                                                         indices.data(),
                                                         sizeof(indices),
                                                         cudaMemcpyHostToDevice),
                                               "copy SBT indices");
            !result) {
            return result;
        }

        CUdeviceptr centerPointer = centerBuffer->get();
        CUdeviceptr radiusPointer = radiusBuffer->get();

        unsigned int sphereInputFlags[] = {
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        };
        OptixBuildInput sphereInput = {.type = OPTIX_BUILD_INPUT_TYPE_SPHERES,
                                       .sphereArray = {
                                           .vertexBuffers = &centerPointer,
                                           .vertexStrideInBytes = sizeof(float3),
                                           .numVertices = sphereCenter.size(),
                                           .radiusBuffers = &radiusPointer,
                                           .radiusStrideInBytes = sizeof(float),
                                           .singleRadius = false,
                                           .flags = sphereInputFlags,
                                           .numSbtRecords = indices.size(),
                                           .sbtIndexOffsetBuffer = sbtIndexBuffer->get(),
                                           .sbtIndexOffsetSizeInBytes = sizeof(int),
                                           .sbtIndexOffsetStrideInBytes = 0,
                                           .primitiveIndexOffset = 0,
                                       }};

        auto gasHandle = ray_tracer->createAccelBuffer(accelBuildOption, sphereInput);
        if (!gasHandle) {
            return std::unexpected(std::move(gasHandle.error()));
        }
        m_gasHandle = *gasHandle;

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
        const OptixResult moduleResult = optixModuleCreate(ray_tracer->getOptixContext(),
                                                           &moduleCompileOptions,
                                                           &pipelineCompileOptions,
                                                           optixIR.data(),
                                                           optixIR.size(),
                                                           log,
                                                           &log_length,
                                                           &module);
        venusaur::OptixModuleHandle moduleOwner{module};
        if (auto result = venusaur::checkOptix(
                moduleResult, "optixModuleCreate", std::string_view(log, log_length));
            !result) {
            return result;
        }

        OptixModule sphereModuleIS = nullptr;
        OptixBuiltinISOptions sphereISOptions = {
            .builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE,
            .usesMotionBlur = false,
            .buildFlags = OPTIX_BUILD_FLAG_NONE,
            .curveEndcapFlags = OPTIX_CURVE_ENDCAP_DEFAULT,
        };
        const OptixResult builtinResult = optixBuiltinISModuleGet(ray_tracer->getOptixContext(),
                                                                  &moduleCompileOptions,
                                                                  &pipelineCompileOptions,
                                                                  &sphereISOptions,
                                                                  &sphereModuleIS);
        venusaur::OptixModuleHandle sphereModuleOwner{sphereModuleIS};
        if (auto result = venusaur::checkOptix(builtinResult, "optixBuiltinISModuleGet"); !result) {
            return result;
        }

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
        std::array<venusaur::OptixProgramGroupHandle, 4> programGroupOwners;
        {
            char log[2048];
            size_t log_length = sizeof(log);
            const OptixResult groupResult = optixProgramGroupCreate(ray_tracer->getOptixContext(),
                                                                     programGroupDesc.data(),
                                                                     programGroupDesc.size(),
                                                                     &programGroupOptions,
                                                                     log,
                                                                     &log_length,
                                                                     programGroup.array.data());
            for (std::size_t index = 0; index < programGroup.array.size(); ++index) {
                programGroupOwners[index].reset(programGroup.array[index]);
            }
            if (auto result = venusaur::checkOptix(
                    groupResult, "optixProgramGroupCreate", std::string_view(log, log_length));
                !result) {
                return result;
            }
        }

        OptixPipelineLinkOptions pipelineLinkOptions = {
            .maxTraceDepth = 1,
            .maxContinuationCallableDepth = 0,
            .maxDirectCallableDepthFromState = 0,
            .maxDirectCallableDepthFromTraversal = 0,
            .maxTraversableGraphDepth = 0,
        };

        if (auto result = ray_tracer->setupPipeline(pipelineCompileOptions, pipelineLinkOptions, programGroup.array);
            !result) {
            return result;
        }

        auto raygenRecordBuffer = venusaur::allocateDeviceBuffer(sizeof(RayGenSbtRecord));
        if (!raygenRecordBuffer) {
            return std::unexpected(std::move(raygenRecordBuffer.error()));
        }
        RayGenSbtRecord raygenRecord = {};
        if (auto result = venusaur::checkOptix(optixSbtRecordPackHeader(programGroup.raygen, &raygenRecord),
                                               "pack raygen SBT record");
            !result) {
            return result;
        }
        if (auto result = venusaur::checkCuda(cudaMemcpy(reinterpret_cast<void*>(raygenRecordBuffer->get()),
                                                         &raygenRecord,
                                                         sizeof(RayGenSbtRecord),
                                                         cudaMemcpyHostToDevice),
                                               "copy raygen SBT record");
            !result) {
            return result;
        }

        auto missRecordBuffer = venusaur::allocateDeviceBuffer(sizeof(MissSbtRecord));
        if (!missRecordBuffer) {
            return std::unexpected(std::move(missRecordBuffer.error()));
        }
        MissSbtRecord missRecord = {};
        if (auto result = venusaur::checkOptix(optixSbtRecordPackHeader(programGroup.miss, &missRecord),
                                               "pack miss SBT record");
            !result) {
            return result;
        }
        if (auto result = venusaur::checkCuda(cudaMemcpy(reinterpret_cast<void*>(missRecordBuffer->get()),
                                                         &missRecord,
                                                         sizeof(MissSbtRecord),
                                                         cudaMemcpyHostToDevice),
                                               "copy miss SBT record");
            !result) {
            return result;
        }

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

        for (std::size_t index = 0; index < hitGroupRecords.size(); ++index) {
            const OptixProgramGroup group = index < 2 ? programGroup.hitgroup_lambertian : programGroup.hitgroup_metal;
            if (auto result = venusaur::checkOptix(optixSbtRecordPackHeader(group, hitGroupRecords[index].header),
                                                   "pack hitgroup SBT record");
                !result) {
                return result;
            }
        }

        auto hitgroupRecordBuffer = venusaur::allocateDeviceBuffer(sizeof(hitGroupRecords));
        if (!hitgroupRecordBuffer) {
            return std::unexpected(std::move(hitgroupRecordBuffer.error()));
        }
        if (auto result = venusaur::checkCuda(cudaMemcpy(reinterpret_cast<void*>(hitgroupRecordBuffer->get()),
                                                         hitGroupRecords.data(),
                                                         sizeof(hitGroupRecords),
                                                         cudaMemcpyHostToDevice),
                                               "copy hitgroup SBT records");
            !result) {
            return result;
        }

        ray_tracer->setupShaderBindingTable({
            .missRecordStrideInBytes = sizeof(MissSbtRecord),
            .missRecordCount = 1,
            .hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord),
            .hitgroupRecordCount = hitGroupRecords.size(),
            .callablesRecordBase = NULL,
            .callablesRecordStrideInBytes = 0,
            .callablesRecordCount = 0,
        },
                                            std::move(*raygenRecordBuffer),
                                            std::move(*missRecordBuffer),
                                            std::move(*hitgroupRecordBuffer));

        if (auto result = ray_tracer->allocateParams(sizeof(MetalParams)); !result) {
            return result;
        }

        return {};
    }

    typedef venusaur::RayTracer::SbtRecord<void> RayGenSbtRecord;
    typedef venusaur::RayTracer::SbtRecord<void> MissSbtRecord;
    typedef venusaur::RayTracer::SbtRecord<Material> HitGroupSbtRecord;

    OptixTraversableHandle m_gasHandle = 0;

    unsigned int m_subframeIndex = 0;

    MetalParams m_params;

    venusaur::Result<std::span<const std::byte>> setupParams(uchar4* image, uint32_t width, uint32_t height) {
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
