add_rules("mode.debug", "mode.release")

set_languages("c++20")

if is_plat("windows") then
    set_toolchains("clang-cl")
    add_defines("NOMINMAX")
elseif is_plat("linux") then
    set_toolchains("clang")
end 

package("glfw")
    add_deps("cmake")
    set_sourcedir(path.join(os.scriptdir(), "third_party/glfw"))
    on_install(function (package)
        local configs = {}
        table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
        table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
        table.insert(configs, "-DGLFW_BUILD_DOCS=OFF")
        table.insert(configs, "-DGLFW_BUILD_EXAMPLES=OFF")
        table.insert(configs, "-DGLFW_BUILD_TESTS=OFF")
        if is_plat("windows") then
            table.insert(configs, "-DGLFW_BUILD_WIN32=ON")
            table.insert(configs, "-DUSE_MSVC_RUNTIME_LIBRARY_DLL=ON")
        end
        import("package.tools.cmake").install(package, configs)
    end)
package_end()

add_requires("glfw")

target("core")
    set_kind("static")
    add_files("core/*.cpp")
    add_includedirs("core/include", {public = true})
    if is_plat("windows") then
        add_syslinks("gdi32")
    end
    -- glfw
    add_packages("glfw", {public = true})
    -- gl3w
    add_includedirs("third_party/gl3w/include", {public = true})
    add_files("third_party/gl3w/src/gl3w.c")
    -- spdlog
    add_includedirs("third_party/spdlog", {public = true})
    -- imgui
    add_includedirs("third_party/imgui", "third_party/imgui/backends")
    add_files("third_party/imgui/*.cpp", "third_party/imgui/backends/*.cpp")
    -- glm
    add_includedirs("third_party/glm", {public = true})
    -- cuda
    local cuda_path = os.getenv("CUDA_PATH")
    if cuda_path then
        add_includedirs(cuda_path .. "/include", {public = true})
        add_linkdirs(cuda_path .. "/lib/x64")
        add_links("cudart_static", "nvrtc")
    else
        print("Warning: CUDA_PATH environment variable is not set. CUDA include directory will not be added.")
    end
    -- optix
    if is_plat("windows") then
        add_includedirs("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.0.0\\include", {public = true})
    end

target("rtow")
    set_kind("binary")
    add_files("rtow/*.cpp")
    add_deps("core")

    after_build(function (target)
        os.cp("rtow/*.cu", "$(builddir)/$(plat)/$(arch)/$(mode)")
    end)

