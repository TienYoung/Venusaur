# Agent 工作说明

> 本文件是 AI agent（Claude Code、Codex 等）进入本仓库时的固定入口。用户不会每次重复项目背景，所有上下文都在下列文档中。

## 开工前必读（按顺序）

1. [`PROJECT_STATUS.md`](PROJECT_STATUS.md)：当前里程碑、最近验证、已知 blocker 和**唯一下一步**。这是事实来源，先读它。
2. `git status --short --branch`：确认工作树与 STATUS 中记录的预期一致。
3. 按需查阅：
   - [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)：长期架构、设计约束、目标架构、历史与分支地图、推荐重构顺序。
   - [`PROJECT_QUESTIONS.md`](PROJECT_QUESTIONS.md)：“感觉不对但未批准实施”的设计疑问。只在相关任务开始前讨论，不要顺手实施。
   - [`docs/notes/`](docs/notes/)：设计学习笔记（按编号）。动到相关代码前先读对应笔记；用户识别出新的典型问题时，按同样格式新增一篇。

如果用户只说“继续”或没有给出具体任务，就从 STATUS 的“唯一下一步”开始，并先向用户确认。

## 项目一句话

Venusaur 是基于 CUDA + OptiX 9.1 的 *Ray Tracing in One Weekend* 实现。当前 `Reconstruction` 分支在做架构重构（不是加功能）：Result 错误模型、RAII 资源所有权、IoC，之后再恢复渲染功能。

## 不可违背的约束

- **错误处理：** 可预期失败统一用 `Result<T> = std::expected<T, Error>` 向上传，像 Rust `?` 一样 early return；不要新增 `throw`、`exit`、assert 作为错误路径。只由 `main` 记录错误并返回失败码。
- **Rule of Zero：** 业务/编排 class 不手写 destructor/copy/move；资源释放只存在于窄小的 RAII handle/deleter（如 `UniqueResource`）中。析构不 throw。
- **平坦流程：** 长而线性、对称的 create/shutdown 流程优于为缩短代码而拆出的 manager/factory/继承层。封装必须闭合一项真实责任（所有权、析构顺序、不变量）。
- **IoC：** `main` 是 composition root；Application 最终只依赖小型 renderer capability。Microsoft Proxy 不是目标本身。
- **日志与字符串：** 日志参数直接交给 `spdlog::*`；非日志字符串用 `std::format`。不用 `std::cout`/`std::cerr`/`fmt::format`。
- **范围：** 不扩成通用引擎；重构阶段不加材质/功能。

## 构建与工具（Windows only）

需要环境变量 `CUDA_PATH`（CUDA 13）与 `OPTIX_INSTALL_DIR`（OptiX SDK 9.1），工具链 clang-cl 22 + xmake。

```powershell
xmake f -m release
xmake build rtow
xmake project -k compile_commands --lsp=clangd .vscode   # clangd 读取 .vscode/compile_commands.json
xmake run rtow
```

## 代码约定

- 修改项目自有 C/C++ 后格式化并验收（排除 `third_party/` 与 legacy 教程文件）：
  ```powershell
  clang-format -i --style=file --fallback-style=none <changed-files>
  clang-format --dry-run --Werror --style=file --fallback-style=none <changed-files>
  ```
- 分析架构时忽略 `third_party/`（占 tracked 文件绝大多数）。
- legacy/dead 文件清单见 GUIDE §3.1，不要把它们当作现行 API。

## 交接协议

1. 开始一个里程碑时，先在 STATUS 写明 `in_progress`、目标和验收条件。
2. 完成时更新 STATUS：结果、验证命令与结论、唯一下一步；代码与文档一起提交。
3. 稳定的架构结论移入 GUIDE；未决疑问进 QUESTIONS。
4. 中途停止时保留 `in_progress` 和明确的恢复动作——不要用聊天上下文代替项目状态。
5. 每个里程碑本地提交一次（Conventional Commits 风格），不自动 push。
