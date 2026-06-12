# 标注速查（外包版）

为每个任务写一条 JSON 标注，告诉评测系统这个任务应该怎么验证。

## 一句话原则

**先问：这个任务最终能不能用一段文字答案交差？**

- **能** → 走答案侧 **Type 1 / 2 / 3**（三选一）
- **不能，必须看到页面动作或页面状态本身才算完成** → 走 **Type 4**
- 任务失效、网站打不开、要求真实购买/发邮件等不可评测的动作 → 标 **UA**

`Search for / Find / What is / 筛选后告诉我...` 这类任务，**不管筛选条件多复杂**，最终都能给出文字答案，必须落在 1/2/3，不要写成 4。
`Open XXX page / Download XXX / Add to cart / Subscribe / Sign up` 这类必须看到页面动作的任务，才用 4。

## 5 种类型一览

| 类型 | 什么时候用 | 必填字段 |
|---:|---|---|
| 1 | 答案稳定不变，可以直接写出标准事实 | `answer_facts` |
| 2 | 答案以网页当前内容为准（最新/排名/状态），一两个网页就能验证 | `urls` |
| 3 | 答案开放、可接受答案很多，或需要看完成过程证据判断真实性 | `answer_points`（可加 `reason`） |
| 4 | 纯操作任务，或答案无法证明页面状态时额外加上 | `page_states`（每条带 `timing`） |
| UA | 任务失效或无法公平评测 | `reason` |

## JSON 模板

```json
{
  "task_id": "任务 ID（必须和原任务一致）",
  "types": [1],              // 数组；答案侧 1/2/3 只能选一种；纯操作题写 [4]；UA 写 ["UA"]
  "answer_facts": [...],     // 含 1 时填
  "urls": [...],             // 含 2 时填
  "answer_points": [...],    // 含 3 时填
  "page_states": [...],      // 含 4 时填，每条要写 timing: "final" 或 "intermediate"
  "reason": "...",           // 含 3 时可选；UA 时必填
  "notes": "..."             // 可选：任选其一 / 满足其中几条等特殊规则写这里
}
```

**禁止组合**：`[1,2]`、`[1,3]`、`[2,3]`、`["UA", 任何其他]`。
**允许组合**：`[1,4]`、`[2,4]`、`[3,4]`（混合任务，见最后一节）。

## 判断流程（按顺序问自己）

1. 任务失效 / 看不懂 / 要求真实购买发邮件 → `["UA"]` + `reason`。
2. 任务最终能不能用文字答案交差？
   - **不能**（Open / Download / Add to cart / Subscribe...）→ 只写 `[4]`，填 `page_states`。
   - **能** → 继续往下走答案侧。
3. 答案能不能直接写出稳定事实？→ 选 `1`，填 `answer_facts`。
4. 答案随网页当前内容变化，但一两个网页就能验证？→ 选 `2`，填 `urls`。
5. 答案开放、可接受多个、或需要看轨迹证据判断真实性？→ 选 `3`，填 `answer_points`。
6. 答案侧选完后，**只有当文字答案无法证明必须打开的页面或页面状态时**，再额外加 `4`。

## 每种类型一个例子

### Type 1：静态标准答案

任务：`derivative of x^2 when x=5.6`

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

### Type 2：目标 URL

任务：`Check the current standings for MLS on Fox Sports.`

```json
{
  "task_id": "b6d10e9bd19b4009a02dea0e98f4e1ae",
  "types": [2],
  "urls": ["https://www.foxsports.com/soccer/mls/standings"]
}
```

### Type 3：轨迹证据核验

任务：`Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars suitable for 6 people.`

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
  "notes": "可接受食谱可能有多个，最终答案给出的一个食谱满足全部 answer_points 即可。"
}
```

### Type 4：页面状态验证

任务：`Open the page with an overview of the submission of releases on Discogs.`

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

### Type 1 + 4：混合（答案 + 页面状态都要验）

任务：`Search the title 'GPT-4 Technical Report' and access this paper through HTML format. Read the paper on this page and tell me what is 'one of the main goals of developing such models' mentioned in the Introduction.`

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

### UA：无法标注

```json
{
  "task_id": "example-ua-1",
  "types": ["UA"],
  "reason": "目标网页已不可访问，无法确定任务要求的正确答案或可验证状态。"
}
```

## 反例对照（最常见错误）

| 任务 | ❌ 错误标法 | ✅ 正确标法 |
|---|---|---|
| `Search for red shoes under $100, sort by customer rating` | `[4]` 写 page_states 把筛选条件列出来 | `[3]`，`answer_points` 写答案必须满足的条件（价格、颜色、排序结果） |
| `What is the current MLS leader?` | `[1]` 直接写答案 | `[2]`，填官方榜单 URL（榜单会变） |
| `Open the Investor Relations page and download the latest 10-K` | `[3]` 写"轨迹证据要包含下载行为" | `[4]`，page_states 写"打开 Investor Relations 页面 + 触发 10-K 下载" |
| `Add Barbie Dreamhouse to cart and tell me its price` | 只写 `[1]` 商品价格 | `[1, 4]`：价格 + 购物车页面必须包含 Barbie Dreamhouse |
| `Find a vegetarian lasagna recipe with >100 reviews` | `[1]` 写一个固定食谱名 | `[3]`，因为合格食谱有多个 |

## 字段填写要点

### `answer_facts`（Type 1）
- 写事实本身，**不要**写「答案必须包含……」这类验证要求。
- 数字写单位、日期写完整。
- 多个答案任选其一 / 只满足其中几条 → 规则写到 `notes`。

### `urls`（Type 2）
- 直接包含答案的页面，**不要**只填首页。
- 不要填搜索结果页、临时链接、带 session 的链接。
- 优先官方来源。

### `answer_points`（Type 3）
- 写「合格答案应该满足什么条件」（来源、主题、数量、属性、约束）。
- **不要**写解题路径或具体点击步骤。
- 不要把它当作"懒得标"的兜底选项。

### `page_states`（Type 4）
- 写「必须打开什么页面 + 该页面呈现什么状态」，**不要**写具体点击操作。
- 每条都要写 `timing`：
  - `"final"`：任务最终就停在这个状态。
  - `"intermediate"`：过程中必须经过的页面，但最终可能停在别处。
- 涉及真实提交、购买、发邮件等动作，只验证到**提交前**的安全状态。

### `reason`（UA / Type 3）
- UA 时必填，写明为什么无法标注。
- Type 3 时可选，写明为什么不能用 Type 1 或 Type 2。

## 提交前自查

- [ ] JSON 能正常解析？
- [ ] `task_id` 和原任务一致？
- [ ] `types` 只包含 `1` / `2` / `3` / `4` 或单独的 `"UA"`？
- [ ] 答案侧 1、2、3 只选了一种？没有 `[1,2]` / `[1,3]` / `[2,3]`？
- [ ] `"UA"` 没和其他类型混用？
- [ ] 含 `1` → `answer_facts` 非空？
- [ ] 含 `2` → `urls` 非空？
- [ ] 含 `3` → `answer_points` 非空，且不是写操作路径？
- [ ] 含 `4` → `page_states` 非空，每条都有 `timing`（`"final"` 或 `"intermediate"`）？
- [ ] 含 `4` 时，确认了文字答案无法证明页面状态？
- [ ] 任选其一 / 满足其中几条 / 至少 N 个 这类规则写到 `notes` 了？
- [ ] 没参考已有 agent 答案、没绑定具体解题路径？

## 通用原则

1. 只看任务本身怎么评测，**不看**当前已有回答是否做对。
2. 不要写「必须点击哪里」「必须滚到哪里」「必须搜哪个关键词」，标注要对所有解题路径公平。
3. 信息查找类：重点标"标准事实"「验证要点」「证据网页」。
4. 操作类：重点标"必须打开什么页面 + 必须呈现什么状态"。
5. 任何会产生外部影响的动作（真实购买、发帖、发邮件、预约）→ 只标到提交前的状态，或直接 `UA`。
6. 模型/搜索引擎只能辅助找答案，**最终标注必须以打开目标网页核验为准**。

完整版规则、字段细节、更多边界情况请看 `ANNOTATION_INSTRUCTIONS.md`。
