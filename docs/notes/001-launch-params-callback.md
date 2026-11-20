# 笔记 001：launch params 回调——“中间有个洞”的设计问题

> 状态：问题已识别，**尚未修复**（计划见 [`PROJECT_GUIDE.md`](../../PROJECT_GUIDE.md) §9 第 6 步）。
> 涉及代码：[`ray_tracer.hpp`](../../core/include/venusaur/ray_tracer.hpp) 的 `SetupParamsCallback`、[`ray_tracer.cpp`](../../core/src/ray_tracer.cpp) 的 `RayTracer::render()`、[`metal_renderer.hpp`](../../RTOW/metal_renderer.hpp) 的 `setupParams()`。
> 引入：`f81546a`（2025-12，Decoupled renderer and ray tracer）；M2 `e668f3f` 只做了安全加固，结构没变。

## 1. 现象

每一帧的渲染流程是：

```text
map PBO -> 得到 image 指针
填 launch params（需要 image 指针 + 相机 + GAS handle + 帧号 ...）
params 拷到 GPU
optixLaunch
unmap PBO
```

- 第 1、3、4、5 步**每个 renderer 都一样**，所以放进了共享的 `RayTracer::render()`。
- 第 2 步**每个 renderer 都不一样**：params 的类型、相机、场景都属于具体 renderer。
- 第 2 步又**依赖第 1 步的结果**，必须插在流程中间。

当时的解法是让 `RayTracer` 持有一个回调，在流程中间“回头调用”具体 renderer：

```cpp
// RayTracer（共享部分）
using SetupParamsCallback =
    std::function<Result<std::span<const std::byte>>(uchar4* image, uint32_t width, uint32_t height)>;

Result<void> RayTracer::render(...) {
    auto mapping = map(...);
    auto params  = m_setupParams(mapping.image, w, h);   // <- 洞
    copy(params); launch(); unmap(mapping);
}
```

能跑，但用起来始终别扭：设置 params 的时机不好掌控，加新功能时也说不清改动该放在哪一边。

## 2. 这到底是什么问题

它不是单独违反了某一条原则，而是几个经典概念同时命中。按“最贴切”到“相关”排列：

### 2.1 Hole in the Middle（中间有个洞）

这是最准确的名字：**公共的开头和结尾夹着一段可变的中间**。map/unmap、open/close、lock/unlock 都是这种形状。

常见的补洞方式有三种：

| 方式 | 谁控制流程 | 本项目 |
|---|---|---|
| 回调 / 高阶函数：把中间那段传进去 | 共享代码 | 当前做法 |
| 作用域 / RAII：开头返回一个 guard，结尾由析构完成 | 调用方 | Q-001 讨论的方向之一 |
| 拆成库函数：调用方自己按顺序调用每一步 | 调用方 | 计划方向 |

### 2.2 Template Method 模式与“好莱坞原则”

`RayTracer::render()` 实际上就是 **Template Method**：父类固定算法骨架，子类填某一步的钩子。这里没用继承，换成了 `std::function`，本质却一样。只是换了写法的继承仍然是继承。

它体现的是 IoC 的**好莱坞原则**：“别打给我们，我们会打给你。”
这对应 Martin Fowler 区分的 **框架（framework）和库（library）**：

- **库**：你的代码调用它，流程由你掌控。
- **框架**：它调用你的代码，流程由它掌控。

`RayTracer` 不知不觉成了一个“框架”，而 `MetalRenderer` 被迫成了插件。问题在于 renderer 恰恰是**最会变化、最需要掌控流程**的那一方：以后的 accumulation、相机移动、resize 后重置，都要在这一帧里插入新步骤。

> 注意：项目确定的目标是 **IoC 在 composition root（`main` 组装依赖）**，不是**把每帧的控制流交给框架**。这两件事都叫“控制反转”，却是不同层级的东西。这里就是 IoC 用错了层级。

### 2.3 SOLID 视角

| 原则 | 是否命中 | 说明 |
|---|---|---|
| **O**pen/Closed（开闭） | **最直接** | 扩展点（回调签名）开得太窄，只给了 `image, width, height`。renderer 想要更多帧信息时，只能**修改** `RayTracer` 的公共接口，做不到只**扩展**。 |
| **S**ingle Responsibility（单一职责） | 是 | `RayTracer` 同时是设备上下文、资源仓库、pipeline 构建器和每帧编排者，有很多理由要改它。 |
| **D**ependency Inversion（依赖倒置） | 部分 | DIP 要求抽象由**高层策略**定义。现在回调签名由底层的共享流程定义，高层的 renderer 只能迁就它，抽象的归属方向反了。 |
| **I**nterface Segregation、**L**iskov | 基本不相关 | 不要硬套。 |

所以“这违反了 SOLID 哪一条？”最贴切的回答是 **开闭原则**，根源则是 **单一职责**。不过 SOLID 只能描述症状，2.1 和 2.2 才说出了问题的形状。

### 2.4 时序耦合与 connascence（共生性）

用来精确描述“哪里在暗中相互依赖”：

- **时序耦合（temporal coupling）：** `allocateParams()`、`setRenderCallback()` 必须在 `render()` 之前调用，而 `main` 里的 `renderer` 变量必须一直存活。这些顺序和生命周期都没有体现在类型里，只能靠运行时报错发现。
- **大小共生（connascence of type/size）：** `allocateParams(sizeof(MetalParams))` 和回调返回的 `span` 大小，在两个地方各自声明同一个事实。
- **数据所有权分裂：** params 的**内容**归 `MetalRenderer`，params 的 **GPU buffer** 归 `RayTracer`。同一份数据被劈成两半，没有任何一方完整拥有它。

## 3. 怎么认出这类问题（症状清单）

以后遇到下面这些感觉，可以怀疑是同一类问题：

- “这个 **时机** 有点麻烦”：可变部分需要流程中途才产生的数据。
- 名叫 `XxxRenderer` 的类却**没有 `render()`**，只往别处“安装”状态和回调。
- 每加一个功能都要**改共享类的回调签名**，或者往里塞越来越多参数。
- 读具体实现时，**看不出自己的代码什么时候被调用**。
- 需要 `weak_ptr` 或“对象已不存在”这类检查，来保护一个本该很简单的调用。
- 同一份数据的大小或类型要在两处保持一致。

## 4. 怎么解

### 计划方向：把框架改回库（翻转控制）

```text
MetalPathTracer::render(surface, frame) -> Result<void>:
    mapping  = surface.map()                                   // 开头：显式
    m_params = { mapping.image, camera, frame.index, ... }     // 中间：时机一目了然
    launch(device, m_pipeline, m_sbt, m_paramsBuffer, m_params, w, h)  // 共享部分只是普通函数
    surface.unmap(mapping)                                     // 结尾：显式
```

- 共享的东西仍然共享，只是从“回调你的骨架”变成“你调用的函数”。
- params 的类型、GPU buffer 和生命周期全部归 renderer 自己，所有权分裂和大小共生一起消失。
- 流程在每个 renderer 里会**重复约 4 行显式调用**。这是有意接受的代价，符合项目“平坦流程优于框架层”的原则（GUIDE §8.1）。

### 被否决或不推荐的替代方案

- **加宽回调签名（传一个 `FrameContext` 结构体）**：能缓解开闭问题，但仍是框架形态，时机和所有权问题不变。只适合当作过渡。
- **改用继承或 CRTP 的 Template Method**：只换了语法，本质与现状相同。
- **再加一层 manager/pipeline 抽象**：违反项目的平坦流程原则。

## 5. 什么时候回调是合适的

回调本身没有错。可变部分**小、稳定、不需要掌控流程**时，回调就是正确选择。例如 OptiX 的 `logCallbackFunction`、GLFW 的 key callback：它们只是“通知你一下”，并不参与决定流程怎么走。

判断标准：**谁最可能改变这段流程，就让谁拥有这段流程。**

## 6. 延伸阅读关键词

- *Hole in the middle pattern*
- *Template Method pattern*；*Hollywood Principle*
- Martin Fowler, *Inversion of Control*（framework vs library）
- *Connascence*（Meilir Page-Jones；Jim Weirich 的演讲 *The Grand Unified Theory of Software Design*）
- *Temporal coupling*
- SOLID：*Open/Closed Principle*、*Single Responsibility Principle*
