# Venusaur 项目状态

> 新对话从这里开始。本文件只保存当前状态、最近验证和唯一下一步；长期架构与历史见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)，尚未决定是否实施的疑问见 [`PROJECT_QUESTIONS.md`](PROJECT_QUESTIONS.md)，更早过程由 Git 历史保存。

## 当前快照

- 更新时间：2026-08-14（America/Toronto）
- 分支：`Reconstruction`
- 代码基线：`b0f5d89`（M2.3）
- 当前里程碑：`M2.5`，`complete`
- 工作树预期：里程碑提交后 clean；只允许存在 ignored build/cache 产物
- 发布策略：每个完成的里程碑本地提交一次，不自动 push

## 当前里程碑：M2.5 待讨论设计队列

目标：建立与已批准路线分离的待讨论表，记录感觉不对但今天不实施的设计问题。

验收结果：

- 独立表包含 RenderTarget Mapping 对称性、RayTracer `create()`/`initialize()` 分段与 `exception.hpp` 来源/混合错误模型三项。
- 每项都只记录当前事实、不适感与决策前必须回答的问题，明确标记为“未批准实施”。
- 今天不修改任何 C/C++ 代码，M3 的已批准目标不变。

## 已确认的长期架构约束

- **IoC/依赖注入：** `main` 作为 composition root，Application 最终只依赖小型 renderer capability。Microsoft Proxy 可以重新评估，但不是 IoC 的必要条件。
- **Result Pattern：** host 已选择 C++23 `std::expected<T, Error>`；不要长期混用 Result、异常、`exit` 和 assert。
- **Rule of Zero：** 业务与编排 class 通过组合窄小 RAII handle 获得自然的 special members；清理逻辑只存在于底层 handle/deleter。
- M1 的异常边界与显式删除 `Application` copy/move 是安全过渡，不代表最终设计已经满足以上约束。

## 里程碑路线

| 里程碑 | 状态 | 结果/目标 |
|---|---|---|
| M0 清理与进度基线 | complete | 旧草稿已丢弃；指南和状态入口已建立 |
| M1 构建基线与 Application 生命周期 | complete | 默认构建恢复；Application 地址稳定，支持部分失败清理与正确 teardown |
| M2 底层资源 RAII | complete | C++23 Result + active handle RAII + mapped PBO guard |
| M2.1 日志与 clangd | complete | 日志入口统一到 spdlog；clangd 接入 xmake compile database；普通字符串误用 bundled fmt 由 M2.2 纠正 |
| M2.2 格式化边界修正 | complete | spdlog 只负责日志；非日志字符串使用 `std::format` |
| M2.3 格式与交接基线 | complete | active 源码 clang-format 基线；今日总结与当前/目标架构图 |
| M2.4 平坦流程与错误边界决策 | complete | 记录显式线性编排、资源封装与致命失败清理策略 |
| M2.5 待讨论设计队列 | complete | 独立记录 Mapping、RayTracer initialize 和 `exception.hpp`；不承诺实施 |
| M3 Application Result 与线性生命周期 | pending | 消除 active Result -> `throw`/`exit`；保持 GLFW/ImGui 创建/销毁显式对称 |
| M4 Active/legacy 边界 | pending | 整理失效教程源码、构建目标与路径大小写 |

## 本轮进度总结（2026-08-13 至 2026-08-14）

- M0：审计并丢弃有悬空 callback、泄漏和错误模型缺口的旧 `expected + RAII` 草稿；建立指南和状态入口。
- M1：移除未完成依赖反转且阻塞 Clang 22 的 Microsoft Proxy 接入；稳定 Application 地址、GLFW/ImGui teardown 和默认构建。IoC 仍是目标，Proxy 不是目标本身。
- 架构意图：确认 composition-root IoC、C++23 `Result<T> = std::expected<T, Error>` 与业务 class Rule of Zero。
- M2：以 `UniqueResource` 覆盖 active GL/CUDA/OptiX handle，建立部分构造安全、无抛 cleanup、mapped PBO guard 和 weak renderer callback。
- M2.1–M2.2：日志直接交给 spdlog；非日志字符串使用 `std::format`；clangd 改为读取 xmake compilation database。
- M2.3：建立 active 源码 clang-format 22.1.3 基线与强制验收协议，并固化今日总结与当前/目标架构图。
- M2.4：确认长而平坦的显式流程优于为缩短代码而封装；资源所有权仍由窄 RAII 类型闭合；Application 错误上传提前为 M3。
- M2.5：建立待讨论表，停放 Mapping 生命期、无独立责任的 initialize 与混合来源 `exception.hpp` 三项疑问，今天不作实现决定。

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
- `2026-08-14`（M2.1）：`xmake project -k compile_commands --lsp=clangd build` 生成 ignored compilation database；`.clangd` 自动加载成功。
- `2026-08-14`（M2.1）：VS clangd 22.1.3 检查 `RTOW/main.cpp` 与 `core/src/render_target.cpp`，均为 0 errors。
- `2026-08-14`（M2.1）：`xmake build -v rtow` 成功；active host 日志扫描和 `git diff --check` 通过。
- `2026-08-14`（M2.2）：非 third-party 源码不再包含 `spdlog/fmt`、`fmt::format`、`std::cout` 或 `std::cerr`；六处 `std::format` 均用于非日志字符串。
- `2026-08-14`（M2.2）：`xmake build -v rtow` 成功；clangd 对 `RTOW/main.cpp` 以及 Application/RayTracer 修改行检查均为 0 errors；`git diff --check` 通过。
- clangd 整文件 `--check` Application 时会在内置 ExtractFunction 动作上报 3 个 break/continue 提取错误；AST 构建、源码诊断和修改行检查正常，不是编译错误。
- `2026-08-14`（M2.3）：VS LLVM clang-format 22.1.3 对 active 自有源码建立基线，随后 `--dry-run --Werror` 通过；`third_party` 和 legacy 未改写。
- `2026-08-14`（M2.3）：`xmake build -v rtow` 成功；clangd 对 `RTOW/main.cpp` 与 `core/src/render_target.cpp` 均为 0 errors；`git diff --check` 通过。
- `2026-08-14`（M2.4）：仅更新决策与交接文档，未修改代码；`git diff --check` 通过，沿用 M2.3 的构建/clangd/clang-format 验证基线。
- `2026-08-14`（M2.5）：只读核对三项当前调用形态；仅新增/更新 Markdown，未修改代码。

## 已知 blocker

- M3 无外部 blocker。当前 `Application::State` 和 `Application::run()` 会把底层 Result 转为 `runtime_error`，NVRTC 宏则直接 `exit`。
- `build/compile_commands.json` 是 ignored 生成物；configure、toolchain 或 include 改变后需重新生成。NVRTC `.cu` 的独立 LSP 支持尚未建立。
- M4 的已知范围：active 与 legacy 文件仍混在 `RTOW`/`core` 中，xmake 的 `rtow` 路径大小写只在 Windows 上偶然可用。
- M2 未自动运行交互式 GUI；启动、渲染、窗口关闭和真实 teardown 仍需 smoke test 覆盖。

## 唯一下一步

开始 M3：让可预期失败像 Rust `?` 一样显式向上传。将 Application 的 fallible 创建与 `run()` 改为 Result 边界，将 NVRTC 编译从 `exit` 改为 Result，最终只由 `main` 记录错误并返回失败码。实现保持长而平坦、early-return 的线性流程；不新增继承、manager/factory 层或致命启动失败的复杂 rollback。GLFW/ImGui 正常关闭仍显式逆序 shutdown，callback 必须指向地址稳定的状态。

## 交接协议

1. 开始工作前读取本文件和 `git status --short --branch`。
2. 开始一个里程碑时，先写明 `in_progress`、目标和验收条件。
3. 完成时记录结果、验证和唯一下一步；代码与文档一起提交。
4. 生成或修改自有 C/C++ 源码后，使用仓库 `.clang-format` 格式化变更文件，并以 `--dry-run --Werror` 验收；排除 third-party 和 legacy。
5. 若中途停止，保留 `in_progress` 状态和明确的恢复动作，不用聊天上下文代替项目状态。
