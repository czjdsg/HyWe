# 标注说明

为每个任务写一条 JSON 标注，说明这个任务后续应该怎么评测。

每个任务的答案侧只能在 Type 1、Type 2、Type 3 里选一种，优先级是 1 > 2 > 3。纯操作类型、不要求最终答案的任务只标 Type 4。需要答案的任务中，只有当最终答案本身不能证明页面状态时，才额外标注 Type 4。如果任务无法标注，则只标 `UA`。

> **重要：Type 1/2/3 与 Type 4 的本质区别在于「任务最终结果能不能用一段文字答案提供出来」。**
>
> - 只要任务最终能写出一段答案（无论是事实、网页内容，还是开放结果），无论筛选条件、查询参数多复杂，都应该标成 Type 1、2、3 中的一种。比如 `Search for ...`、`Find ...`、`What is ...`、`筛选后告诉我...` 这类任务，即使要叠多层筛选条件、跨多个页面，最终都能给出文字答案，必须落在答案侧。
> - 只有那些**必须通过网页操作才能算完成、且没有可以替代的文字答案**的任务，才标 Type 4。比如 `Open the XXX page`、`Download the XXX file`、`Subscribe to ...`、`Add ... to cart`、`Sign up for ...` 等以打开页面、达到页面状态、产生页面动作为唯一目标的任务。
> - 判断时先问自己：「这个任务能不能用一段答案文字交差？」能就走 1/2/3；不能、必须看到页面状态本身才算完成，才走 4。

当前使用 4 种评测要素，另有一个特殊标记 `UA`：

| 类型编号 | 方式 | 什么时候标 | 需要填写什么 |
|---:|---|---|---|
| 1 | 静态标准答案 | 答案稳定不变，可以直接写出标准事实或标准答案 | `answer_facts` |
| 2 | 目标 URL | 答案以网页当前内容为准，可以从一个或几个网页验证 | `urls` |
| 3 | 轨迹证据核验 | 答案没有唯一标准，或需要看完成过程中的证据判断是否真实 | `answer_points`，`reason` 可选 |
| 4 | 页面状态验证 | 纯操作任务，或任务要求中途打开特定页面/达到特定页面状态且最终答案无法证明该状态 | `page_states` |
| UA | 无法标注 | 任务失效、网站不可用、任务描述不足，或无法设计公平评测 | `reason` |

## 任务提供的信息

标注时会给你一批任务。每个任务通常包含这些字段：

- `task_id`：任务唯一 ID。写标注 JSON 时必须原样复制。
- `dataset`：任务来自哪个数据集，例如 `Online-Mind2Web` 或 `WebVoyager`。只用于理解来源，标注 JSON 里可以不写。
- `question`：任务要求。判断标注类型和填写内容时，以这个字段为准。
- `web`：目标网站或起始网址。你需要打开这个网站，查看任务对应页面或信息后再标注。
- `domain`：网站域名或站点名称。辅助理解任务来源。
- `source_file` / `line_index`：原始任务文件位置。一般不用填写到标注结果里，只用于追溯。

标注 JSON 最少只需要写 `task_id`、`types` 和对应标注字段。不要把 `question`、`web`、`dataset` 等任务原始字段复制进最终标注，除非你自己为了备注临时保留。

## 标注流程

1. 阅读 `question`，确认任务最终是回答问题、验证网页内容、完成网页状态，还是无法标注。
2. 打开 `web` 指向的目标网站，在目标网站上查找任务相关页面或信息。
3. 如果能得到稳定事实，标 Type 1，并把事实写进 `answer_facts`。
4. 如果答案会随网页当前内容变化，但可以由目标网页验证，标 Type 2，并把直接证据页面写进 `urls`。
5. 如果答案开放、可接受结果很多，或者需要结合完成过程证据判断真实性，标 Type 3，并写 `answer_points`。
6. 如果任务是纯操作类型、不要求最终答案，只标 Type 4，并写 `page_states`。
7. 如果任务要求中途打开特定页面或达成页面状态，而且最终答案不能证明该状态，可以在答案侧类型之外额外标 Type 4，并写 `page_states`。
8. 如果目标网页失效、任务无法理解、需要不安全外部动作，或无法设计公平验证，标 `UA`。

标注时应尽量从目标网站本身确认信息。不要只凭常识、搜索摘要或已有 agent 回答来写标注。

## 可以使用的辅助工具

标注人员可以使用 GPT、Claude 等网页端先进模型辅助标注，也可以打开它们的网页搜索功能。

推荐用法：

1. 把 `question` 和 `web` 提供给模型，让模型先尝试找出答案、目标页面或可验证依据。
2. 要求模型给出它使用的网页 URL、页面标题和关键证据。
3. 标注者自己打开目标网页或模型给出的候选 URL，核对答案和证据是否真实存在。
4. 核对通过后，再写最终 JSON 标注。

注意：

- 模型输出只能作为辅助，不能直接当作最终标注。
- 最终标注必须以目标网页或可打开的证据网页为准。
- 如果模型给出的 URL 打不开、内容不匹配、来自非目标网站，不能直接采用。
- 如果模型给出了答案但你无法在网页中验证，通常不要标 Type 1 或 Type 2；应考虑 Type 3 或 `UA`。
- 如果多个模型结论不一致，以你实际打开网页核验到的内容为准。

## JSON 总格式

建议写成一个 JSON 数组，每个任务一条。每条用 `types` 表示这个任务需要哪些评测要素；只有一种也写成数组。

```json
[
  {
    "task_id": "任务 ID",
    "types": [1],
    "answer_facts": [
      "标准事实或答案 1",
      "标准事实或答案 2"
    ]
  }
]
```

每条都必须有：

- `task_id`：任务 ID，必须和原任务里的 ID 一致。
- `types`：数组。答案侧只能包含 `1`、`2`、`3` 中的一种；纯操作题可以只写 `[4]`；需要答案时，`4` 可以按需额外加入；如果任务无法标注，则只写 `["UA"]`。

根据 `types` 里包含的类型，再填写对应字段：

- 包含 `1` 时，填 `answer_facts`。
- 包含 `2` 时，填 `urls`。
- 包含 `3` 时，填 `answer_points`，可选填 `reason`。
- 包含 `4` 时，填 `page_states`。每条 `page_states` 都要写自己的 `timing`。
- 如果是 `["UA"]`，填 `reason`，说明为什么任务无法标注。

可选字段：

- `notes`：备注。用于说明特殊判定逻辑，比如多个答案任选其一、列出的事实只需满足其中几条、允许的等价说法、数量要求或例外情况。

如果一个任务确实需要答案侧和页面状态两个维度验证，就在同一条里同时填写两个维度的字段。例如最终答案无法证明页面状态是否达成时，可以写 `types: [1, 4]`，同时填写 `answer_facts` 和 `page_states`。

如果一个任务是纯操作类型，不要求输出最终答案，只写 `types: [4]`，不要为了凑答案侧类型去选 `1`、`2` 或 `3`。

不要同时选择 `1`、`2`、`3`。例如不要写 `types: [1, 2]`、`types: [2, 3]` 或 `types: [1, 3]`。

`UA` 不能和其他类型混用。不要写 `types: ["UA", 1]`。

## 总体原则

1. 只看任务本身应该怎么评测，不看当前已有回答有没有做对。
2. 优先覆盖任务必须验证的维度，不要为了省事漏掉答案或页面状态。
3. 标注要避免偏向某一种解题路线，不要写“必须搜索”“必须点击第几个按钮”“必须滚动到哪里”。
4. 信息查找类任务，重点标注“标准事实是什么”“答案应该满足什么验证要点”或“哪个网页能证明答案”。
5. 如果必须标页面状态，重点标注“必须打开哪个页面、该页面应呈现什么状态”，不要记录每一步点击。
6. 不要要求真实提交、购买、预约、发帖、发邮件等会产生外部影响的动作。只标注安全的、可观察的完成状态。

## 快速选择

先问：任务是否仍然有效、可访问、可理解、可设计公平评测？

如果任务失效或无法标注，写 `types: ["UA"]`，并填写 `reason`。

如果任务可标注，再判断任务最终是否需要输出答案。判断标准只有一个：**最终结果能不能用一段答案文字提供出来。**

- 能用文字回答（包括 `Search for ...`、`Find ...`、`What is ...`、`筛选 / 排序后给出结果` 等查询、检索、筛选类任务）：走答案侧 Type 1/2/3。哪怕筛选条件再复杂，只要最终能给出答案，都不应该标 Type 4。
- 不能用文字回答，必须真的打开页面、达到页面状态、产生页面动作（比如 `Open XXX page`、`Download XXX`、`Add to cart`、`Subscribe`）：走 Type 4。

如果需要输出答案，先判断答案侧。Type 1、Type 2、Type 3 只能选一种：

1. 答案是否固定不变，并且你可以直接写出标准事实或标准答案？

是：包含 `1`，不要再选 `2` 或 `3`。

2. 答案是否可能变化，但可以从一个或几个网页直接验证？

是：包含 `2`，不要再选 `3`。

3. 答案是否开放、可接受答案很多、没有单一网页能完全证明，或者必须结合完成过程中的证据才能判断？

是：包含 `3`。

如果任务既要求网页操作，又要求回答信息，继续判断最终答案能不能覆盖页面状态结果。能覆盖时，只标答案侧；不能覆盖时，才同时包含 `4`。

如果任务不要求输出答案，而是只要求完成网页状态变化，只包含 `4`。

如果你觉得任务需要交互才能得到答案，请在 `2` 和 `3` 之间判断：能用网页证据验证就选 `2`，否则选 `3`。

## Type 1：静态标准答案

适合：

- 答案不会随时间变化。
- 答案可以写成明确事实，例如名称、数字、日期、标题、代码、列表项或固定结论。
- 后续只需要判断最终回答是否包含这些事实。

需要填写：

- `answer_facts`：数组，每个元素是一条标准事实或标准答案。

填写规则：

- 每个 `answer_facts` 默认都是必须被最终答案覆盖的事实。
- 写事实本身，不要写“答案必须包含……”这类验证要求。
- 如果有多个可接受说法，可以在同一条里写明等价表达。
- 如果多个答案任选其一，或者列出的事实只需要满足其中几条，不要把规则混进 `answer_facts`；把判定逻辑写到 `notes`。
- 数字要写单位或允许误差。
- 日期要写完整。
- 列表任务要写出具体可接受的列表项或明确的固定答案。
- 不要写解题过程。

示例：

```json
{
  "task_id": "example-static-1",
  "types": [1],
  "answer_facts": [
    "The derivative of x^2 is 2x.",
    "When x = 5.6, the derivative value is 11.2."
  ]
}
```

如果任务要求的是特定答案，而不是任意可接受答案，可以写：

```json
{
  "task_id": "example-static-2",
  "types": [1],
  "answer_facts": [
    "The Sega Genesis debuted in North America at US$189.99.",
    "The console was typically bundled with Altered Beast."
  ]
}
```

如果有多个可接受答案，或者只需要满足其中几条，写成：

```json
{
  "task_id": "example-static-3",
  "types": [1],
  "answer_facts": [
    "iCourts – Centre of Excellence for International Courts and Governance",
    "MOBILE – Center of Excellence for Global Mobility Law",
    "CLIMA – Centre for Climate Change Law and Governance",
    "CeBIL – Centre for Advanced Studies in Biomedical Innovation Law"
  ],
  "notes": "任务只要求列出三个研究中心；最终答案只要包含 answer_facts 中任意三个即可。"
}
```

不适合：

- “最新”“当前”“今天”等会变的答案。
- 必须以网页当前内容为准的答案。
- 可接受答案太多，无法合理写成标准事实。

## Type 2：目标 URL

适合：

- 任务是回答问题。
- 答案可能会变化，或者必须以某个网页当前内容为准。
- 你能给出一个或几个网页，后续评测可以用这些网页判断答案是否正确。

需要填写：

- `urls`：数组，每个元素是一个网页 URL。

填写规则：

- 填直接包含答案的页面，不要只填网站首页。
- 如果答案由多个页面共同决定，可以填多个 URL。
- 对“最新文章”“当前状态”“排行榜”等任务，填展示该信息的列表页或详情页。
- 优先填官方来源。
- 不要填搜索结果页、临时链接、带登录状态或 session 的链接。
- `urls` 里只写 URL，不写解释。

示例：

```json
{
  "task_id": "example-url-1",
  "types": [2],
  "urls": [
    "https://www.thehindu.com/sci-tech/energy-and-environment/"
  ]
}
```

```json
{
  "task_id": "example-url-2",
  "types": [2],
  "urls": [
    "https://blog.collegevine.com/what-is-northwesterns-acceptance-rate-and-admissions-requirements"
  ]
}
```

不适合：

- 答案需要复杂交互才能出现，单靠网页内容看不到。
- 没有一个或几个网页能证明答案。
- 任务目标是操作页面，而不是回答信息。

## Type 3：轨迹证据核验

适合：

- 任务是回答问题。
- 没有简洁的标准答案。
- 没有一个或几个网页能单独证明答案。
- 可接受答案很多，比如“找一个例子”“任选一篇文章”“总结主要内容”。
- 需要根据完成过程中实际查到、读到、看到的证据来判断最终答案是否真实。

需要填写：

- `answer_points`：数组，每个元素是一条最终答案必须满足、并且需要被轨迹证据支持的验证要点。
- 可选填 `reason`：一句话说明为什么不能简单写标准答案或目标 URL。

填写规则：

- 不要写唯一标准答案，除非任务本身要求最终答案必须包含某个固定要素。
- 要写“合格答案应该满足什么条件”，例如来源、主题、数量、属性、约束条件。
- 每个 `answer_points` 都应该能通过轨迹中的网页内容、搜索结果、访问页面或其他工具输出找到支持证据。
- 如果多个条件只需满足其中几条，或多个答案任选其一，把规则写到 `notes`，不要塞进单条 `answer_points`。
- 不要写具体操作路径。
- 不要把它当作“不知道怎么标”的默认选项。只有前两类确实不合适时才选。

示例：

```json
{
  "task_id": "example-evidence-1",
  "types": [3],
  "answer_points": [
    "答案必须给出一个具体文章标题或页面标题。",
    "答案必须说明该文章或页面对应的 business type。",
    "轨迹证据必须支持该文章或页面确实来自任务要求的网站或栏目。"
  ],
  "reason": "可接受文章和 business type 很多，没有唯一标准答案，需要根据完成过程中找到的文章证据判断最终回答是否真实。"
}
```

```json
{
  "task_id": "example-evidence-2",
  "types": [3],
  "answer_points": [
    "答案必须直接回应任务问题，而不是只描述搜索过程。",
    "答案中的关键事实必须能在轨迹证据中找到支持。",
    "如果答案列出多个项目，每个项目都必须满足任务给出的限制条件。"
  ],
  "reason": "答案可能来自多个页面片段，需要结合完成过程中的证据判断是否真实。"
}
```

不适合：

- 能直接写出稳定答案，应该包含 `1`。
- 能给出直接证明网页，应该包含 `2`。
- 任务本质是打开页面或达成页面状态，且最终答案无法证明该页面状态，才应该包含 `4`。

## Type 4：页面状态验证

> **核心判定**：只有当「任务最终结果无法用一段文字答案提供出来」时，才用 Type 4。`Search for / Find / What is / 筛选后告诉我...` 这类任务，无论条件多复杂，最终都能给出答案，必须落在 1/2/3，不要用 4。Type 4 是为 `Open XXX page`、`Download XXX`、`Subscribe / Add to cart / Sign up` 这种「必须看到页面动作或状态才算完成」的任务准备的。

适合：

- 任务是纯操作类型，不要求输出最终答案，且文字答案无法替代页面动作。典型动词：`Open`、`Download`、`Subscribe`、`Add ... to cart`、`Sign up`、`Log in`、`Bookmark`、`Follow`。
- 任务要求中途打开某个页面或达到某个可观察的页面状态，且最终答案不能证明该状态。
- 仅仅回答文字不能算完成任务。
- 成功可以拆成几个“必须打开的页面及其对应状态”。

需要填写：

- `page_states`：数组，每个元素是一条必须出现过的页面及其对应状态。
- 每个 `page_states` 元素推荐写成对象，包含 `timing` 和 `state`。
- `timing` 只能填 `"final"` 或 `"intermediate"`。

填写规则：

- 写“必须打开什么页面”和“该页面上必须呈现什么状态”，不要写“具体点击哪里”。
- 如果任务最终就停在目标页面或目标状态，该条页面状态填 `"timing": "final"`。
- 如果任务需要在过程中打开某些证据页面，但最终可能停在别的页面或最终答案里，该条页面状态填 `"timing": "intermediate"`。
- 一个任务可以同时包含 final 和 intermediate 两种页面状态，要分别写在不同的 `page_states` 元素里。
- 每个页面状态都应该能通过 URL、页面标题、可见文本、表单值或截图判断。
- 所有页面状态都应该是必要的。
- 不要过细，不要把每次点击、滚动都写出来。
- 不要过泛，不要只写“完成任务”。
- 如果涉及提交、购买、预约等真实外部动作，只要求到提交前的安全状态。

示例（任务：`Open the page with an overview of the submission of releases on Discogs.`）：

```json
{
  "task_id": "example-operation-1",
  "types": [4],
  "page_states": [
    {
      "timing": "final",
      "state": "打开 Discogs 支持站点上的 Overview Of Submission Guidelines For Releases 页面，URL 为 https://support.discogs.com/hc/en-us/articles/360004016474-Overview-Of-Submission-Guidelines-For-Releases。"
    }
  ]
}
```

示例（任务：`Download the latest annual report (PDF) from the Investor Relations page.`）：

```json
{
  "task_id": "example-operation-2",
  "types": [4],
  "page_states": [
    {
      "timing": "intermediate",
      "state": "必须打开 Investor Relations 页面，页面列出可下载的 annual report 资源。"
    },
    {
      "timing": "final",
      "state": "必须触发最新一份 annual report PDF 的下载动作，或停留在该 PDF 已开始下载/已打开的页面状态。"
    }
  ]
}
```

> **反例（不要这样标）**：任务 `Search for red shoes under $100, sort by customer rating` 看上去要求多层筛选，但最终能输出「排序后第一条/合格列表」作为文字答案，应该走 Type 2 或 Type 3，不要写成 Type 4 的 page_states。

不适合：

- 任务只是让回答一个信息，即使涉及筛选、排序、跨页查询，只要最终能给出文字答案，就不应该用 Type 4。
- 页面操作只是查找信息的手段，不是最终目标。
- `Search for ...`、`Find ...`、`What is ...`、`筛选 / 排序后告诉我...` 等查询、检索类任务，应走 Type 1/2/3。

## UA：无法标注

适合：

- 原任务依赖的网站或页面已经失效，无法判断目标内容。
- 任务描述缺失关键信息，无法知道应该评测什么。
- 任务要求登录、真实购买、真实预约、发帖、发送邮件等不适合评测的外部动作。
- 任务目标本身不清楚，无法设计对所有模式都公平的验证方式。
- 任务所需信息不可访问，且不能通过合理的公开网页或轨迹证据验证。

需要填写：

- `types`: 只能写 `["UA"]`。
- `reason`：一句话说明为什么无法标注。

示例：

```json
{
  "task_id": "example-ua-1",
  "types": ["UA"],
  "reason": "目标网页已经不可访问，无法确定任务要求的正确答案或可验证状态。"
}
```

不适合：

- 只是答案比较难查。难查不等于无法标注。
- 可以写标准事实、目标 URL、轨迹证据核验要点或页面状态中的任意一种。
- 当前某个 agent 没做出来。标注只看任务本身，不看已有结果。

## 混合任务

有些任务既要打开或达到特定页面状态，又要输出答案。这种情况下不要默认两边都标。先判断页面状态结果是否已经体现在答案中。

如果任务是纯操作类型，不要求输出最终答案，它不是混合任务，只标 `types: [4]`。

如果最终答案已经能证明任务完成，只标答案侧，不标页面状态。例如任务要求“筛选后告诉我最便宜商品的名称和价格”，只要答案侧标注能要求名称、价格和筛选条件对应关系，并且这些信息能证明筛选结果，就不需要再标 `4`。

如果最终答案不能证明页面状态是否达成，才额外标页面状态。例如任务要求“把符合条件的商品加入购物车，并告诉我商品名称”，答案里的商品名称不能证明购物车状态，因此需要同时标答案和购物车页面状态。

需要组合标注的常见情况：

- `types: [1, 4]`：既要验证最终答案是否包含固定事实，也要验证页面状态是否达成。需要同时填写 `answer_facts` 和 `page_states`。
- `types: [2, 4]`：答案需要以网页当前内容为准，同时还要验证页面状态是否达成。
- `types: [3, 4]`：答案需要结合轨迹证据判断真实性，同时还要验证页面状态是否达成。需要同时填写 `answer_points` 和 `page_states`。

示例：

```json
{
  "task_id": "example-mixed-1",
  "types": [1, 4],
  "answer_facts": [
    "加入购物车的目标商品是 Barbie Dreamhouse。"
  ],
  "page_states": [
    "必须打开购物车页面，页面显示目标商品 Barbie Dreamhouse 已在购物车中。",
    "购物车页面中商品规格与任务要求一致。"
  ]
}
```

## 常见情况

### “最新 / 当前 / today / current”

通常包含 `2`，填写当前信息所在网页。

如果没有可靠网页，或者必须根据完成过程里的多个证据判断，包含 `3`。

### “找任意一个例子”

不要随便写一个固定标准答案。

如果目标网页能证明任意合格答案，包含 `2`。

如果需要看实际找到了哪个例子以及证据是否支持，包含 `3`。

### 原任务里有 steps 或 criteria

可以参考，但不要照抄路径。

坏例子：

```text
Successfully navigates to the search page.
Clicks the latest article.
```

好例子：

```text
答案来自目标网站上对应主题的页面。
答案给出的标题必须是该栏目当前最新文章标题。
```

### 静态答案和 URL 都可以

如果答案稳定、短小、不会随页面变化，包含 `1`。

如果答案必须以网页当前内容为准，包含 `2`。

### 页面状态结果能写在答案里

如果任务要求通过操作找出一个结果，并把这个结果作为答案输出，通常只标答案侧。**`Search for / Find / 筛选 / 排序`这类检索类任务，无论筛选条件多复杂，最终都能给出文字答案，应该走 Type 1/2/3，不要写成 Type 4。**

例如“筛选酒店后告诉我评分最高的酒店名称和价格”，如果答案是固定事实，用 `answer_facts` 写酒店名称和价格；如果合格答案可能变化或有多个，用 `answer_points` 写答案必须满足的筛选条件。只要答案侧足以验证任务完成，就不需要标页面状态。

只有当答案无法证明页面状态时才标 Type 4。Type 4 的典型场景是 `Open the XXX page`、`Download XXX file`、`Add ... to cart`、`Subscribe / Sign up`、`Bookmark` 等必须看到页面动作或状态的任务。例如“把这个商品加入购物车并告诉我商品名”，商品名不能证明加入购物车成功，所以需要标 `4`。

## 提交前检查

- JSON 是否能被正常解析？
- 每条是否都有 `task_id`？
- 每条是否都有 `types`？
- `types` 是否只包含一个答案侧类型，也就是 `1`、`2`、`3` 中最多一个？
- 纯操作类型、不要求最终答案的任务是否只写了 `types: [4]`？
- 如果 `types` 包含 `4`，是否还保持答案侧最多只选一种？
- `types` 是否没有出现 `[1, 2]`、`[1, 3]`、`[2, 3]` 这类错误组合？
- `types` 是否只包含合法值 `1`、`2`、`3`、`4`，或单独的 `"UA"`？
- 如果是 `"UA"`，是否没有和其他类型混用？
- 如果是 `"UA"`，是否填写了无法标注的 `reason`？
- 包含 `1` 时，是否填写了非空 `answer_facts`？
- 包含 `2` 时，是否填写了非空 `urls`？
- 包含 `3` 时，是否填写了非空 `answer_points`？
- 包含 `3` 时，`answer_points` 是否写的是答案验证要点，而不是具体操作路径？
- 包含 `4` 时，是否填写了非空 `page_states`？
- 包含 `4` 时，每条 `page_states` 是否都填写了 `timing`，且只能是 `"final"` 或 `"intermediate"`？
- 如果存在“任选其一”“满足其中几条”“至少 N 个”等特殊规则，是否写进了 `notes`？
- 如果标了 `4`，是否确认最终答案本身不能证明页面状态已经达成？
- `page_states` 是否写清了必须打开的页面及其对应状态，而不是点击路径？
- 标注是否没有依赖某一种具体解题路线？
- 标注是否没有参考当前已有回答是否做对？

## 推荐判断顺序

1. 如果任务失效或无法设计公平标注，写 `["UA"]`。
2. **先问：任务最终结果能不能用一段文字答案提供出来？**
   - **能**（包括 `Search for ...`、`Find ...`、`What is ...`、`筛选 / 排序后告诉我...`，无论筛选条件多复杂）：走答案侧 1/2/3，跳到第 4 步。
   - **不能**（任务必须打开页面、下载文件、加入购物车、订阅等，无法用文字答案替代）：只写 `[4]`。
3. 纯操作型任务（`Open / Download / Subscribe / Add to cart` 等）只写 `[4]`，不要为了凑答案侧再选 1/2/3。
4. 答案侧 `1`、`2`、`3` 只能选一种：
   - 能用 Type 1 就不用 Type 2；能写稳定标准事实或标准答案时，选 `1`。
   - 不能用 Type 1，但能用目标网页验证答案时，选 `2`。
   - 不能用 Type 1 或 Type 2，且需要根据轨迹证据判断真实性时，选 `3`。
5. 选完答案侧后，只有最终答案无法证明必须打开的页面或页面状态是否达成时，才额外包含 `4`。

如果页面状态结果已经能被最终答案验证，不要标 `4`。

## Case 示例

下面五个任务覆盖当前使用的五种标注情况。示例里的空数组或占位文本需要标注者按任务实际内容补全。

### Case 1：Type 1 静态标准答案

原任务：

```text
derivative of x^2 when x=5.6
```

任务信息：

```json
{
  "dataset": "WebVoyager",
  "task_id": "Wolfram_Alpha--0",
  "web": "https://www.wolframalpha.com/"
}
```

建议标注：

```json
{
  "task_id": "Wolfram_Alpha--0",
  "types": [1],
  "answer_facts": [
    "The derivative of x^2 is 2x.",
    "When x = 5.6, the derivative value is 11.2."
  ]
}
```

### Case 2：Type 2 目标 URL

原任务：

```text
Check the current standings for MLS on Fox Sports.
```

任务信息：

```json
{
  "dataset": "Online-Mind2Web",
  "task_id": "b6d10e9bd19b4009a02dea0e98f4e1ae",
  "web": "https://www.foxsports.com/"
}
```

建议标注：

```json
{
  "task_id": "b6d10e9bd19b4009a02dea0e98f4e1ae",
  "types": [2],
  "urls": [
    "https://www.foxsports.com/soccer/mls/standings"
  ]
}
```

### Case 3：Type 3 轨迹证据核验

原任务：

```text
Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars suitable for 6 people.
```

任务信息：

```json
{
  "dataset": "WebVoyager",
  "task_id": "Allrecipes--0",
  "web": "https://www.allrecipes.com/"
}
```

建议标注：

```json
{
  "task_id": "Allrecipes--0",
  "types": [3],
  "answer_points": [
    "答案必须给出一个具体的 Allrecipes 食谱名称。",
    "该食谱必须是 vegetarian lasagna 或明确符合素食千层面主题。",
    "该食谱评论数超过 100。",
    "该食谱评分至少为 4.5 星。",
    "该食谱适合 6 人份，或答案说明如何调整为 6 人份。"
  ],
  "notes": "可接受的食谱可能有多个；只要最终答案给出的一个食谱满足全部 answer_points 即可。"
}
```

### Case 4：Type 4 页面状态验证

原任务：

```text
Open the page with an overview of the submission of releases on Discogs.
```

任务信息：

```json
{
  "dataset": "Online-Mind2Web",
  "task_id": "fb7b4f784cfde003e2548fdf4e8d6b4f",
  "web": "https://www.discogs.com/"
}
```

建议标注：

```json
{
  "task_id": "fb7b4f784cfde003e2548fdf4e8d6b4f",
  "types": [4],
  "page_states": [
    {
      "timing": "final",
      "state": "打开 Discogs 支持站点上的 Overview Of Submission Guidelines For Releases 页面，URL 为 https://support.discogs.com/hc/en-us/articles/360004016474-Overview-Of-Submission-Guidelines-For-Releases。"
    }
  ]
}
```

### Case 5：混合类型验证，Type 1 + Type 4

原任务：

```text
Search the title 'GPT-4 Technical Report' and access this paper through HTML format. Read the paper on this page and tell me what is 'one of the main goals of developing such models' mentioned in the Introduction.
```

任务信息：

```json
{
  "dataset": "WebVoyager",
  "task_id": "ArXiv--39",
  "web": "https://arxiv.org/"
}
```

为什么是混合类型：

这个任务有稳定标准事实答案，同时明确要求通过 HTML format 访问论文页面。最终答案里即使写出了正确内容，也不能证明 agent 真的打开过 HTML 版论文页面，所以需要同时验证标准事实和中间页面状态。

建议标注：

```json
{
  "task_id": "ArXiv--39",
  "types": [1, 4],
  "answer_facts": [
    "One of the main goals of developing such models is to improve their ability to understand and generate natural language text, particularly in more complex and nuanced scenarios."
  ],
  "page_states": [
    {
      "timing": "intermediate",
      "state": "打开 GPT-4 Technical Report 的 HTML 版论文页面，URL 为 https://arxiv.org/html/2303.08774v6。"
    }
  ]
}
```
