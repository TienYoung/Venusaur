# Venusaur 项目状态

> 新对话从这里开始。本文件只保存当前状态、最近验证和唯一下一步；长期架构与历史见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)，更早过程由 Git 历史保存。

## 当前快照

- 更新时间：2026-08-14（America/Toronto）
- 分支：`Reconstruction`
- 代码基线：本文件所在提交（M2；M1 为 `40cc3ea`）
- 当前里程碑：`M2`，`complete`
- 工作树预期：里程碑提交后 clean；只允许存在 ignored build/cache 产物
- 发布策略：每个完成的里程碑本地提交一次，不自动 push

## 当前里程碑：M2 底层资源 RAII

目标：让 active GL/CUDA/OptiX handle 具有唯一所有者、部分构造安全和无抛析构，并让上层 class 向 Rule of Zero 收敛。

实际结果：

- 主机代码升级到 C++23，项目 Result 采用 `std::expected` 别名，不新增 expected 依赖；NVRTC device source 暂留 C++20。
- 新建通用 move-only `UniqueResource` 作为叶子 RAII 基础设施；业务/编排 class 不手写 destructor/copy/move。
- active 范围只包括 `RenderTarget`、`Rasterizer`、`RayTracer`、`MetalRenderer` 及 mapped PBO guard；旧教程 renderer 留给 M3。
- setup/create/render 的可预期 CUDA/OptiX/GL 失败改为 `Result`；deleter 只做 best-effort cleanup 和诊断，绝不 throw/terminate。
- 不在 M2 翻转 RayTracer/MetalRenderer 职责，不恢复材质、相机或其他 RTOW 功能。
- RenderTarget/Rasterizer/RayTracer/MetalRenderer 不再声明 destructor/copy/move；所有 active owner 由通用 `UniqueResource` 或标准智能指针表达。
- mapped PBO guard 覆盖 render 失败路径；renderer callback 使用 `weak_ptr`，SBT/GAS/params setup 检查配置和容量。
- OptiX function table definition 从公共 header 移到唯一 `ray_tracer.cpp`。

## 已确认的长期架构约束

- **IoC/依赖注入：** `main` 作为 composition root，Application 最终只依赖小型 renderer capability。Microsoft Proxy 可以重新评估，但不是 IoC 的必要条件。
- **Result Pattern：** host 已选择 C++23 `std::expected<T, Error>`；不要长期混用 Result、异常、`exit` 和 assert。
- **Rule of Zero：** 业务与编排 class 通过组合窄小 RAII handle 获得自然的 special members；清理逻辑只存在于底层 handle/deleter。
- M1 的异常边界与显式删除 `Application` copy/move 是安全过渡，不代表最终设计已经满足以上约束。

验收条件：

- texture/PBO/GL program/VAO、CUDA graphics registration/stream/buffers、OptiX context/pipeline/module/program group 都由 RAII owner 覆盖。
- 构造或 setup 中途失败不会泄漏已经创建的 active handle；shader 成功 link 后也会释放 shader objects。
- RayTracer 销毁 CUDA stream；RenderTarget 注销 CUDA interop 后删除 PBO/texture；所有 cleanup 路径 `noexcept`。
- render 在 map 后任一步失败都会通过 scoped guard 尝试 unmap。
- 默认 Release 配置完成 `xmake build rtow`，且 `git diff --check` 通过。
- 代码与文档在同一 M2 提交中提交；交互式 GUI 若未运行必须明确记录。

## 里程碑路线

| 里程碑 | 状态 | 结果/目标 |
|---|---|---|
| M0 清理与进度基线 | complete | 旧草稿已丢弃；指南和状态入口已建立 |
| M1 构建基线与 Application 生命周期 | complete | 默认构建恢复；Application 地址稳定，支持部分失败清理与正确 teardown |
| M2 底层资源 RAII | complete | C++23 Result + active handle RAII + mapped PBO guard |
| M3 Active/legacy 边界 | pending | 整理失效教程源码、构建目标与路径大小写 |

## 最近验证

- `2026-08-13`：清理前五文件 diff 与 `fcc286a` 完全一致。
- `2026-08-13`：五个精确路径已恢复到 `HEAD`，未触碰 stash、分支和其他源码。
- `xmake f -m release --cxxflags=`：成功，确认使用默认配置且没有诊断宏。
- `xmake build -v rtow`：失败。Clang 22.1.3 即使在 `-std:c++20` 下也让 vendored Proxy v4 进入 `trivially_relocatable_if_eligible` 分支，并在 `proxy.h:929` 等处产生语法错误。
- `2026-08-13`（M1）：移除 active Proxy 依赖后，`xmake f -m release --cxxflags=` 与 `xmake build -v rtow` 成功，Clang 22.1.3 完成 `rtow.exe` 编译和链接。
- `2026-08-13`（M1）：`git diff --check` 通过；active 源码与 `xmake.lua` 不再引用 Proxy。
- `2026-08-13`（架构意图补录）：确认 IoC、Result Pattern、Rule of Zero 为后续重构约束；M1 实现明确标记为过渡状态。
- `2026-08-14`（M2）：`xmake f -m release --cxxflags=` 成功；clang-cl 为 host 选择 `-std:c++latest`。
- `2026-08-14`（M2）：`xmake build -r -v rtow` 全量成功，完成 `rtow.exe` 编译和链接。
- `2026-08-14`（M2）：`git diff --check`、active raw-owner 扫描、高层 destructor 扫描和 OptiX function-table 单定义检查通过。

## 已知 blocker

- M3 无外部 blocker。active 与 legacy 文件仍混在 `RTOW`/`core` 中，且 xmake 的 `rtow` 路径大小写只在 Windows 上偶然可用。
- Application 仍把底层 Result 转为异常，NVRTC 宏仍会 `exit`；它们是最终 Result Pattern 尚未闭合的边界，但不阻塞 M3。
- M2 未自动运行交互式 GUI；启动、渲染、窗口关闭和真实 teardown 仍需 smoke test 覆盖。

## 唯一下一步

启动 M3：先根据 xmake include/build graph 给每个 `RTOW`/`core` 文件标记 active 或 legacy，再修正 `RTOW` 路径大小写，并选择“移出 active 源码树”或“整理为可独立构建 examples”；不要在该里程碑重写 renderer 架构。

## 交接协议

1. 开始工作前读取本文件和 `git status --short --branch`。
2. 开始一个里程碑时，先写明 `in_progress`、目标和验收条件。
3. 完成时记录结果、验证和唯一下一步；代码与文档一起提交。
4. 若中途停止，保留 `in_progress` 状态和明确的恢复动作，不用聊天上下文代替项目状态。
