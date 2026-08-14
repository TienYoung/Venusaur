# Venusaur 项目状态

> 新对话从这里开始。本文件只保存当前状态、最近验证和唯一下一步；长期架构与历史见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)，更早过程由 Git 历史保存。

## 当前快照

- 更新时间：2026-08-13（America/Toronto）
- 分支：`Reconstruction`
- 代码基线：本文件所在提交（M1；父里程碑 M0 为 `3125a5098a4c0ae48eb9a4b4e7ac0c3202e6a143`）
- 当前里程碑：`M1`，`complete`
- 工作树预期：里程碑提交后 clean；只允许存在 ignored build/cache 产物
- 发布策略：每个完成的里程碑本地提交一次，不自动 push

## 当前里程碑：M1 构建基线与 Application 生命周期

目标：恢复 Clang 22 下的默认 C++20 构建，并让 `Application` 地址稳定、部分构造失败可清理、析构顺序正确。

实际结果：

- 保持 C++20；内部初始化继续用异常，`main` 作为统一错误边界。
- 移除未形成真实可替换性的 Microsoft Proxy；`Application` 暂时直接持有现有 `RayTracer`。
- 保留 `Application(int, int)`、`SetRenderer(...)` 和 `run()`，但显式禁止 copy/move。
- 以成员 RAII 表达 GLFW、window 与 ImGui 生命周期，并用声明顺序保证 GL 资源在 context/window 前释放。
- 以 PIMPL/forward declaration 收窄 `application.hpp` 的依赖面；第二个并存 Application 会明确失败。
- 未配置 renderer 时抛出 `logic_error`，应用入口统一记录未处理的 `std::exception` 并返回失败码。

## 已确认的长期架构约束

- **IoC/依赖注入：** `main` 作为 composition root，Application 最终只依赖小型 renderer capability。Microsoft Proxy 可以重新评估，但不是 IoC 的必要条件。
- **Result Pattern：** 可预期失败最终统一为 `Result<T, E>`；不要长期混用 Result、异常、`exit` 和 assert。C++20/C++23 及具体 Result 实现需在采用前明确。
- **Rule of Zero：** 业务与编排 class 通过组合窄小 RAII handle 获得自然的 special members；清理逻辑只存在于底层 handle/deleter。
- M1 的异常边界与显式删除 `Application` copy/move 是安全过渡，不代表最终设计已经满足以上约束。

验收条件：

- 默认 Release 配置能够完成 `xmake build rtow`，不使用 feature-test macro workaround。
- active 源码和构建配置不再依赖 Microsoft Proxy。
- `Application` 不可 copy/move，GLFW callback 中的 `this` 地址稳定。
- GLFW/window/ImGui 支持部分初始化失败清理；正常析构顺序为 ImGui、GL 对象、window、GLFW runtime。
- 未设置 renderer 时明确失败，顶层异常记录后返回失败码。
- 代码与文档通过 `git diff --check` 并在同一 M1 提交中提交。

## 里程碑路线

| 里程碑 | 状态 | 结果/目标 |
|---|---|---|
| M0 清理与进度基线 | complete | 旧草稿已丢弃；指南和状态入口已建立 |
| M1 构建基线与 Application 生命周期 | complete | 默认构建恢复；Application 地址稳定，支持部分失败清理与正确 teardown |
| M2 底层资源 RAII | pending | 让 GL/CUDA/OptiX handle move-only、部分构造安全、析构不抛 |
| M3 Active/legacy 边界 | pending | 整理失效教程源码、构建目标与路径大小写 |

## 最近验证

- `2026-08-13`：清理前五文件 diff 与 `fcc286a` 完全一致。
- `2026-08-13`：五个精确路径已恢复到 `HEAD`，未触碰 stash、分支和其他源码。
- `xmake f -m release --cxxflags=`：成功，确认使用默认配置且没有诊断宏。
- `xmake build -v rtow`：失败。Clang 22.1.3 即使在 `-std:c++20` 下也让 vendored Proxy v4 进入 `trivially_relocatable_if_eligible` 分支，并在 `proxy.h:929` 等处产生语法错误。
- `2026-08-13`（M1）：移除 active Proxy 依赖后，`xmake f -m release --cxxflags=` 与 `xmake build -v rtow` 成功，Clang 22.1.3 完成 `rtow.exe` 编译和链接。
- `2026-08-13`（M1）：`git diff --check` 通过；active 源码与 `xmake.lua` 不再引用 Proxy。
- `2026-08-13`（架构意图补录）：确认 IoC、Result Pattern、Rule of Zero 为后续重构约束；M1 实现明确标记为过渡状态。

## 已知 blocker

- M2 无外部 blocker。当前主要风险在底层资源：`RenderTarget` 未删除 GL texture/PBO，`RayTracer` 未销毁 CUDA stream，多个 raw handle 可复制/覆盖，析构路径中的检查宏可能抛出。
- 本里程碑只完成 build 验证，未自动运行交互式 GUI；启动、渲染和窗口关闭仍需后续 smoke test 覆盖。

## 唯一下一步

启动 M2：先列出 active GL/CUDA/OptiX handle 及 owner/创建/销毁路径，并先确定统一 Result 的语言版本/实现；随后按依赖顺序把释放集中到窄小、move-safe、析构 `noexcept` 的 RAII handle，使上层 class 向 Rule of Zero 收敛。不要在同一里程碑顺带改 renderer 职责或恢复 RTOW 功能。

## 交接协议

1. 开始工作前读取本文件和 `git status --short --branch`。
2. 开始一个里程碑时，先写明 `in_progress`、目标和验收条件。
3. 完成时记录结果、验证和唯一下一步；代码与文档一起提交。
4. 若中途停止，保留 `in_progress` 状态和明确的恢复动作，不用聊天上下文代替项目状态。
