# HybridWebAgent 架构说明

本文档介绍 `/data/czj/HybridWebAgent` 当前版本的整体框架、核心模块、执行流程、上下文管理方式，以及与 `Mobile-Agent-v3.5` / `IterResearch` 的结合点。

## 1. 设计目标

`HybridWebAgent` 的目标是做一个通用 Web Agent，而不是只做某一种网页任务。

它希望同时覆盖三类任务：

- 信息搜集类任务
- 网页操作类任务
- 既要搜集信息又要交互操作的混合任务

当前实现的基本思路是：

- 用 `Mobile-Agent-v3.5/browser_use` 提供浏览器执行与 SoM 标注能力
- 用 `IterResearch` 风格的 `<report> + 上下文重建` 管理长链路推理
- 用独立的 `visit` 信息读取链路，把“读网页信息”和“在浏览器里操作页面”分开

## 2. 总体结构

项目内的核心文件如下：

- `run_hybrid_web_agent.py`
  - 入口脚本
  - 负责参数解析、代理配置、任务列表构造、并发调度、结果落盘
- `hybrid_agent.py`
  - 主 agent controller
  - 负责 prompt 组装、LLM 调用、响应解析、工具分派、轨迹记录
- `hybrid_runtime.py`
  - Playwright 浏览器运行时
  - 负责页面打开、截图、SoM 生成、点击/输入/滚动/回退等动作
- `hybrid_prompts.py`
  - 系统提示词、工具 schema、不同轮次的 user prompt 模板
- `hybrid_search.py`
  - `search` 工具
  - 通过 Serper 做搜索
- `hybrid_visit.py`
  - `visit` 工具
  - 通过 Jina Reader 风格链路读取网页文本，再走总结模型摘要
- `hybrid_utils.py`
  - JSONL 读写、文本压缩、tag 提取、task_id 清洗等公共工具
- `visualize_trajectory.py`
  - 单个结果文件的 HTML 轨迹可视化
- `visualize_paired_trajectory.py`
  - `with_web / no_web` 成对对比可视化

## 3. 两层架构

### 3.1 上层：决策与上下文管理

上层是 `hybrid_agent.py` 里的 `HybridBrowserAgent`。

它负责：

- 把任务组织成一轮一轮的决策过程
- 每轮向主模型发送结构化 prompt
- 让模型输出：
  - `<report>...</report>`
  - 然后二选一：
    - `<tool_call>...</tool_call>`
    - `<answer>...</answer>`
- 校验输出格式与工具参数
- 调用工具并把结果作为下一轮上下文的一部分

### 3.2 下层：网页执行与观测

下层是 `hybrid_runtime.py` 里的 `HybridPlaywrightComputer`。

它负责：

- 启动 Chromium
- 创建 Playwright context
- 设置 viewport、UA、时区、代理
- 打开初始网页或 `about:blank`
- 生成：
  - 当前 URL
  - 当前 title
  - SoM 截图
  - 原始截图
  - SoM 文本
- 执行动作：
  - `goto`
  - `click`
  - `fill`
  - `select`
  - `scroll`
  - `wait`
  - `go_back`

## 4. 当前的决策范式

当前 agent 不预设任务类型，而是每一轮都显式判断瓶颈（bottleneck）。

系统提示词中要求模型每步都先判断：

- `missing information`
- `missing interaction`
- `goal already completed`

判断顺序不是只看当前页面，而是：

1. 回顾任务本身
2. 回顾上一轮的 `<report>`
3. 回顾上一轮的动作与观测
4. 查看当前浏览器状态
5. 最后再判断当前真正缺什么

这套逻辑来自 `hybrid_prompts.py`，目的是避免模型：

- 只看页面就盲目点
- 只记旧信息而忽略当前页面
- 被固定绑成“只搜”或“只操作”的 agent

## 5. 上下文管理方式

这套框架的核心不是“全量 message 历史回放”，而是“每轮重建上下文”。

当前一轮送入模型的核心内容是：

- 当前日期
- 任务描述
- 工具列表
- 上一轮 `report`
- 上一轮 `tool_call`
- 上一轮 `observation`
- 当前浏览器状态
  - URL
  - title
  - screenshot
  - SoM 文本

也就是说，它保留的是：

- 上一轮的结构化进度摘要
- 最新一轮的网页观测

而不是把所有历史原样塞回去。

这继承了 `IterResearch` 的思路，但做了网页任务适配。

## 6. 首轮与后续轮的区别

### 6.1 有初始网页时

如果输入中显式提供了 `web/url`，浏览器会先打开该页面。

首轮 prompt 会包含：

- 当前 URL
- 当前 title
- 当前 SoM 文本
- 当前截图

### 6.2 没有初始网页时

如果没有提供 `web/url`，浏览器从 `about:blank` 启动。

此时：

- 首轮走 `NO_BROWSER` prompt
- 不给截图
- 不给当前 URL/title/SoM

只有当浏览器真正进入非 `about:blank` 页面后，后续轮次才切到带浏览器状态的 prompt。

## 7. 工具语义

当前可用工具有：

- `search`
- `visit`
- `goto`
- `click`
- `fill`
- `select`
- `scroll`
- `wait`
- `go_back`

它们的语义边界如下。

### 7.1 `search`

用于：

- 查找候选网站
- 查找候选页面
- 获取外部证据

当前通过 Serper 实现，默认返回前若干条搜索结果摘要。

### 7.2 `visit`

`visit` 是“读信息”的工具，不改变浏览器状态。

它的职责是：

- 读取指定 URL 的网页内容
- 针对 `goal` 做信息抽取
- 返回：
  - `Evidence in page`
  - `Summary`

它不负责：

- 打开浏览器页面
- 生成当前浏览器截图
- 触发后续交互

如果模型需要后续交互，应使用 `goto`。

### 7.3 `goto`

`goto` 是“把页面真正打开到浏览器里”的工具。

适合用于：

- 需要页面状态
- 需要截图
- 需要后续点击/输入/滚动

### 7.4 浏览器动作工具

`click / fill / select / scroll / wait / go_back` 只作用在当前浏览器页面。

其中：

- `click / fill / select / labeled scroll` 必须使用当前 SoM 中的数字 label
- 不允许把可见文本当 label 直接传入

在 `hybrid_agent.py` 里，主 agent 会在执行前做一轮 label 校验，避免模型输出无效文本 label。

## 8. SoM 与浏览器观测

当前浏览器观测的核心是 SoM。

每轮 `current_state()` 会尽量生成：

- `trajectory_som/screenshot{step}.png`
- `trajectory/{step}_full_screenshot.png`
- `SoM_list`
- `format_ele_text`

其中：

- 主模型看到的是带 SoM 的截图
- 同时在 prompt 中看到 SoM 文本

### 当前默认设置

- viewport 默认是 `1080 x 1440`
- `--use-css-som` 默认开启
- 浏览器默认 `headless`
- 浏览器默认 `--no-sandbox`
- 任务代理默认走 `http://127.0.0.1:8990`
- `localhost,127.0.0.1` 默认不走代理，保证本地总结模型可直连

## 9. 模型分工

当前框架支持主模型和总结模型分离。

### 9.1 主模型

主模型负责：

- 瓶颈判断
- 工具选择
- 交互决策
- 最终答案生成

它通过 OpenAI-compatible `chat.completions` 调用。

### 9.2 总结模型

总结模型当前主要服务于 `visit`。

它负责：

- 接收网页正文
- 接收与该网页相关的图片输入
- 返回结构化 JSON：
  - `rational`
  - `evidence`
  - `summary`

这样主 agent 不需要直接吃完整长网页内容，而是拿到更压缩的页面信息。

## 10. `visit` 的具体实现

`visit` 当前在 `hybrid_visit.py` 中实现。

### 10.1 主读取路径

默认先走：

- `https://r.jina.ai/{url}`

当前请求头只额外保留：

- `X-Keep-Img-Data-Url: true`

如果配置了 `JINA_API_KEY`，会带 `Authorization`。

### 10.2 图片处理

`visit` 会尝试给总结模型提供页面图片证据。

当前支持两类图片来源：

- Jina 返回中的 inline `data:image/...`
- Jina 内容里解析出的 Markdown 图片 URL

对于第二类：

- 会下载图片
- 过滤非位图格式
- 转成 `data:image/...;base64,...`
- 再作为多模态输入传给总结模型

当前只保留可安全解码的位图图片，避免 `svg` 或坏图把模型请求打崩。

### 10.3 长网页截断

长网页先按 tokenizer 截断，再按字符数兜底：

- `visit_max_webpage_tokens`
- `visit_max_webpage_chars`

这是从 `IterResearch` 风格演化过来的压缩链路。

### 10.4 Fallback

如果 Jina 失败，还会尝试直接请求目标页面做兜底读取。

## 11. 容错策略

当前实现有几类关键容错：

### 11.1 图片容错

如果主模型或总结模型因为图片报错，例如：

- `cannot identify image file`
- `invalid image`
- `unsupported image`

会自动去掉图片后重试一次。

### 11.2 浏览器状态容错

如果：

- SoM 绘制失败
- 截图失败
- 新开标签页跳到空 URL / `chrome-error://`

运行时会降级处理，而不是直接让整条任务崩掉。

### 11.3 任务级容错

每个任务独立运行。

如果单条任务抛异常：

- 会落一个 `task_error`
- 同时尽可能保留已经采集到的部分 `records` 和 `usage`

这样后续排查时不会丢掉中间轨迹。

## 12. 轨迹与输出

每个任务目录下通常会有：

- `meta.json`
- `result.json`
- `trajectory/`
- `trajectory_som/`

其中 `result.json` 会保存：

- `question`
- `prediction`
- `termination`
- `records`
- `usage`
- `error`（如果失败）

`records` 中的每一步会保存：

- 当前 URL / title
- SoM 文本
- 当前截图路径
- prompt 文本
- 模型原始输出
- 解析后的 `report`
- `tool_call`
- `observation`
- `answer`
- `summary_trace`（针对 `visit`）

此外，批量运行时还会生成一个 JSONL 汇总文件，每行对应一个任务的最终结果。

## 13. 当前并发与批量运行方式

入口脚本支持：

- 单任务运行
- JSONL 数据集批量运行
- `--max-workers` 并发调度

执行模型是：

- 每个任务独立 browser session
- 用 `asyncio.Semaphore` 控制任务并发数
- 用锁保证 JSONL 追加写入安全

因此当前已经支持并发批量 benchmark，不再是单任务专用框架。

## 14. 当前框架的特点

这套框架的主要特点是：

- 不预设任务类型，而是逐步判断当前 bottleneck
- 把“读信息”和“进浏览器操作”拆成两条不同工具链
- 主模型与总结模型可以分开部署
- 保留 SoM grounding，减少纯文本网页代理的盲操作
- 保留 `IterResearch` 风格的 report-driven memory，减轻长上下文膨胀

## 15. 当前局限

当前版本仍有一些明确局限：

- 还没有 benchmark 专用登录脚本
- 对极重交互、多身份态网站仍不够强
- `visit` 仍然依赖外部网页读取链路的稳定性
- 浏览器动作仍然是单步原子动作，没有独立 planner / executor 分层
- 大规模结果分析与汇总还依赖额外脚本，而不是统一 benchmark harness

## 16. 一句话总结

`HybridWebAgent` 当前可以理解成：

**一个把 `Mobile-Agent` 的网页执行能力、`IterResearch` 的上下文压缩思想，以及独立的 Jina `visit` 信息读取链路拼接起来的通用 Web Agent。**

它的核心不是固定成“搜索 agent”或“操作 agent”，而是让模型每一步都先判断：当前真正缺的是信息，还是交互，还是任务已经完成。
