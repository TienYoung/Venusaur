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
        add_links("cudart_static")
    else
        print("Warning: CUDA_PATH environment variable is not set. CUDA include directory will not be added.")
    end
    -- optix
    if is_plat("windows") then
        add_includedirs("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.0.0\\include", {public = true})
    end

target("optixir")
    set_rules()
    set_kind("object")
    set_toolchains("cuda")
    add_cuflags("-Wno-deprecated-gpu-targets", "-optix-ir", "-lineinfo", "--use_fast_math")
    add_includedirs("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.0.0\\include", "core/include")
    add_files("rtow/*.cu")

    after_build_file(function (target, sourcefile)
        local objfile = target:objectfile(sourcefile)
        import("core.project.project")
        local core_target = project.target("core")
        if core_target then
            local core_dir = path.directory(core_target:targetfile())
            local output_file = path.join(core_dir, path.basename((path.basename(objfile))) .. ".optixir")
            os.cp(objfile, output_file)
        end
    end)

target("rtow")
    set_kind("binary")
    add_files("rtow/*.cpp")
    add_deps("core")

-- target("OptiX-IR")
--     set_kind("shared")
--     set_toolchains("cuda")
--     set_extension(".optixir")
--     add_files("src/*.cu")
--     if is_plat("windows") then
--         add_includedirs("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.0.0\\include")
--     end

--     add_culdflags("-optix-ir", "--use_fast_mat")

    -- generate SASS code for SM architecture of current host
    -- add_cugencodes("native")

    -- generate PTX code for the virtual architecture to guarantee compatibility
    -- add_cugencodes("compute_35")

    -- generate SASS code for each SM architecture
    -- add_cugencodes("sm_35", "sm_37", "sm_50", "sm_52", "sm_60", "sm_61", "sm_70", "sm_75")

    -- generate PTX code from the highest SM architecture to guarantee forward-compatibility
    -- add_cugencodes("compute_75")
--
-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro definition
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--

