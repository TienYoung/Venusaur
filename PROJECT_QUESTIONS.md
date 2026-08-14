# Venusaur 待讨论设计问题

> 本文件是“感觉不对，但尚未决定是否修改”的停放表，不是 backlog，也不表示已批准实施。新对话先读 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)；只在相关任务开始前讨论本表对应项。

| ID | 范围与当前事实 | 感觉不对的地方 | 需要讨论/核对后才能决定 | 状态 |
|---|---|---|---|---|
| Q-001 | **RenderTarget Mapping**：`map()` 返回同时携带 `image` 和 `ScopedGraphicsMap guard` 的 `Mapping`；`RayTracer::render()` 经过 params/copy/launch 后才调用 `unmap(std::move(mapping))`；`unmap()` 内部再 `guard.release()`。 | map/unmap 在调用点上隔得较远，guard 与显式 `release()` 又把一个本应对称的生命期拆成两层；读者难以从局部确认何时仍处于 mapped 状态。 | 确定 mapped lifetime 应显式对称还是 lexical/RAII；如果自动 unmap，错误如何上传；PBO -> texture upload 应属于 unmap、present 还是独立步骤；在没有对比清楚这些取舍前不改。 | open / 未批准实施 |
| Q-002 | **RayTracer create/initialize**：`RayTracer::create()` 分配 private object，随后只调用一次 private `initialize()` 并传递其 Result；外部不能单独调用 `initialize()`。 | `initialize()` 目前看起来只是为减少 `create()` 长度而提取，没有形成可复用操作、新不变量或更清晰的错误边界。 | 核对 initialize 是否存在独立责任；比较“将线性流程直接留在 create”与“保留分段”的真实阅读/测试收益；同时单独审视 `shared_ptr` 是所有权需求还是工厂写法遗留。 | open / 未批准实施 |
| Q-003 | **`exception.hpp`**：该文件由维基/示例等多个来源复制后经项目 hack，当前同时包含 GL/CUDA/OptiX check macros、`sutil::Exception`、throwing helpers、noexcept cleanup logging，以及直接 `exit` 的 NVRTC/CUDA macros。 | 来源、许可、命名和项目当前 Result 风格混在一起；可能有未使用部分，也可能有 active 必需的 cleanup helper，不应直接整件重写或删除。 | 先盘点每个 macro/helper 的 active call site 与来源/许可；区分应迁移到 Result 的可预期失败、析构期 best-effort 诊断和可删 legacy；M3 可以移除 active NVRTC `exit` 边界，但不在没有这份审计时顺手“整理整个 exception.hpp”。 | open / 未批准实施 |

决定后：若不做，在表中记录理由并关闭；若要做，先转成有验收条件的里程碑，完成后将稳定结论移入 [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md)。
