# Venusaur

基于 CUDA + OptiX 9.1 的 [*Ray Tracing in One Weekend*](https://raytracing.github.io/) GPU 实现，使用 NVRTC 运行时编译 shader，通过 CUDA/OpenGL PBO 互操作实时显示。

## 当前状态

`Reconstruction` 分支正在进行架构重构，当前可运行的是一个硬编码的四球 Lambertian/Metal 场景。旧 `master` 上的随机球场景、dielectric、景深和 accumulation 尚未迁回新架构。

| 文档 | 内容 |
|---|---|
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | 当前里程碑、最近验证、唯一下一步（**从这里开始**） |
| [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) | 架构、设计约束、历史与分支地图、重构路线 |
| [`PROJECT_QUESTIONS.md`](PROJECT_QUESTIONS.md) | 未批准实施的设计疑问 |
| [`docs/notes/`](docs/notes/) | 设计学习笔记：典型问题的分析与取舍 |
| [`AGENTS.md`](AGENTS.md) | AI agent 工作入口与约定 |

## 构建（Windows）

依赖：xmake、clang-cl（Clang 22）、CUDA 13（`CUDA_PATH`）、OptiX SDK 9.1（`OPTIX_INSTALL_DIR`）。

```powershell
xmake f -m release
xmake build rtow
xmake run rtow
```

## 目录

- `core/`：Application、RenderTarget、Rasterizer、RayTracer 等公共框架
- `RTOW/`：示例程序与 `cuda/` 下的 OptiX device 代码（含部分 legacy 教程章节）
- `third_party/`：vendored 依赖（GLFW、ImGui、gl3w、spdlog、glm）
