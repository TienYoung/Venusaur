# Venusaur 项目状态

> 新对话从这里开始。本文件只保存当前状态、最近验证和唯一下一步；长期架构与历史见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)，更早过程由 Git 历史保存。

## 当前快照

- 更新时间：2026-08-14（America/Toronto）
- 分支：`Reconstruction`
- 代码基线：本文件所在提交（M2.1；M2 为 `e668f3f`）
- 当前里程碑：`M2.1`，`complete`
- 工作树预期：里程碑提交后 clean；只允许存在 ignored build/cache 产物
- 发布策略：每个完成的里程碑本地提交一次，不自动 push

## 当前里程碑：M2.1 日志格式化与 clangd

目标：统一 spdlog/fmt 与字符串格式化边界，并让 clangd 使用 xmake 的真实编译数据库正确解析 host 源码。已完成。

已锁定边界：

- 日志参数直接交给 spdlog；需要生成字符串值的路径使用 spdlog bundled fmt，不再混用 `std::format`。
- clangd 以 xmake 生成到 ignored `build/compile_commands.json` 的数据库为事实来源；`.clangd` 不重复硬编码 SDK include。
- 只修 host active 源码诊断；NVRTC `.cu` 的独立语言服务器支持不在本步骤扩展。

## 已确认的长期架构约束

- **IoC/依赖注入：** `main` 作为 composition root，Application 最终只依赖小型 renderer capability。Microsoft Proxy 可以重新评估，但不是 IoC 的必要条件。
- **Result Pattern：** host 已选择 C++23 `std::expected<T, Error>`；不要长期混用 Result、异常、`exit` 和 assert。
- **Rule of Zero：** 业务与编排 class 通过组合窄小 RAII handle 获得自然的 special members；清理逻辑只存在于底层 handle/deleter。
- M1 的异常边界与显式删除 `Application` copy/move 是安全过渡，不代表最终设计已经满足以上约束。

验收结果：

- active host 日志直接调用 spdlog；只在需要字符串值时使用 spdlog bundled `fmt::format`，不再使用 `std::format` 或 iostream 打印日志。
- `.clangd` 只指向 ignored `build/compile_commands.json`，SDK、toolchain、include 和语言标准由 xmake 生成。
- VS clangd 对 `RTOW/main.cpp` 和 `core/src/render_target.cpp` 自动读取数据库，均完成 0-error check。
- 默认 Release `xmake build -v rtow` 成功，`git diff --check` 通过。

## 里程碑路线

| 里程碑 | 状态 | 结果/目标 |
|---|---|---|
| M0 清理与进度基线 | complete | 旧草稿已丢弃；指南和状态入口已建立 |
| M1 构建基线与 Application 生命周期 | complete | 默认构建恢复；Application 地址稳定，支持部分失败清理与正确 teardown |
| M2 底层资源 RAII | complete | C++23 Result + active handle RAII + mapped PBO guard |
| M2.1 日志格式化与 clangd | complete | 日志统一到 spdlog；字符串用 bundled fmt；clangd 接入 xmake compile database |
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
- `2026-08-14`（M2.1）：`xmake project -k compile_commands --lsp=clangd build` 生成 ignored compilation database；`.clangd` 自动加载成功。
- `2026-08-14`（M2.1）：VS clangd 22.1.3 检查 `RTOW/main.cpp` 与 `core/src/render_target.cpp`，均为 0 errors。
- `2026-08-14`（M2.1）：`xmake build -v rtow` 成功；active host 日志扫描和 `git diff --check` 通过。

## 已知 blocker

- M3 无外部 blocker。active 与 legacy 文件仍混在 `RTOW`/`core` 中，且 xmake 的 `rtow` 路径大小写只在 Windows 上偶然可用。
- `build/compile_commands.json` 是 ignored 生成物；configure、toolchain 或 include 改变后需重新生成。NVRTC `.cu` 的独立 LSP 支持尚未建立。
- Application 仍把底层 Result 转为异常，NVRTC 宏仍会 `exit`；它们是最终 Result Pattern 尚未闭合的边界，但不阻塞 M3。
- M2 未自动运行交互式 GUI；启动、渲染、窗口关闭和真实 teardown 仍需 smoke test 覆盖。

## 唯一下一步

开始 M3：声明 active/legacy 边界，先盘点失效教程文件、构建目标与 `RTOW` 路径大小写；开始前先把 M3 标记为 `in_progress` 并锁定去留策略。

## 交接协议

1. 开始工作前读取本文件和 `git status --short --branch`。
2. 开始一个里程碑时，先写明 `in_progress`、目标和验收条件。
3. 完成时记录结果、验证和唯一下一步；代码与文档一起提交。
4. 若中途停止，保留 `in_progress` 状态和明确的恢复动作，不用聊天上下文代替项目状态。
