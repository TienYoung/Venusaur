# Venusaur 项目状态

> 新对话从这里开始。本文件只保存当前状态、最近验证和唯一下一步；长期架构与历史见 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)，更早过程由 Git 历史保存。

## 当前快照

- 更新时间：2026-08-13（America/Toronto）
- 分支：`Reconstruction`
- 代码基线：`667ed24922c3fdd29ff7ee0289f0eb036ecea281`
- 当前里程碑：`M0`，`complete`
- 工作树预期：里程碑提交后 clean；只允许存在 ignored build/cache 产物
- 发布策略：每个完成的里程碑本地提交一次，不自动 push

## 当前里程碑：M0 清理与进度基线

目标：清除已知不安全的 Application/C++23 草稿，建立任何新对话都能直接接手的文档入口。

已完成：

- 将五个草稿文件与 stash index `fcc286a` 逐一核对。
- 只将 `.clangd`、`RTOW/main.cpp`、`application.hpp/.cpp`、`xmake.lua` 恢复到 `667ed24`。
- 保留草稿的设计意图和失败原因，不保留其实现。
- 建立稳定指南与动态状态分离的记录方式。

验收条件：

- 五个旧文件与代码基线一致。
- 默认 Release 配置完成 `rtow` 构建，或将未经扩大修复的原始失败记录在本文件。
- 文档通过 whitespace/diff 检查。
- 两份文档在同一里程碑提交中提交，提交后工作树干净。

## 里程碑路线

| 里程碑 | 状态 | 结果/目标 |
|---|---|---|
| M0 清理与进度基线 | complete | 旧草稿已丢弃；指南和状态入口已建立 |
| M1 构建基线与 Application 生命周期 | pending | 处理 Proxy/Clang blocker；决定 C++20/C++23 和错误策略；修复地址稳定性与 teardown |
| M2 底层资源 RAII | pending | 让 GL/CUDA/OptiX handle move-only、部分构造安全、析构不抛 |
| M3 Active/legacy 边界 | pending | 整理失效教程源码、构建目标与路径大小写 |

## 最近验证

- `2026-08-13`：清理前五文件 diff 与 `fcc286a` 完全一致。
- `2026-08-13`：五个精确路径已恢复到 `HEAD`，未触碰 stash、分支和其他源码。
- `xmake f -m release --cxxflags=`：成功，确认使用默认配置且没有诊断宏。
- `xmake build -v rtow`：失败。Clang 22.1.3 即使在 `-std:c++20` 下也让 vendored Proxy v4 进入 `trivially_relocatable_if_eligible` 分支，并在 `proxy.h:929` 等处产生语法错误。

## 已知 blocker

- 默认 C++20 Release 基线目前不能用 Clang 22.1.3 构建。M1 必须在更新/移除 Proxy、调整受支持工具链或采用正式兼容修复之间作出决定；不要恢复审计用的 feature-test macro workaround。

## 唯一下一步

为 M1 制定决策完整的实施方案：先恢复受支持的默认构建，再重做 Application 生命周期与错误边界。M1 尚未开始。

## 交接协议

1. 开始工作前读取本文件和 `git status --short --branch`。
2. 开始一个里程碑时，先写明 `in_progress`、目标和验收条件。
3. 完成时记录结果、验证和唯一下一步；代码与文档一起提交。
4. 若中途停止，保留 `in_progress` 状态和明确的恢复动作，不用聊天上下文代替项目状态。
