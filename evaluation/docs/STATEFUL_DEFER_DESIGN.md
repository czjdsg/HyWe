# Stateful defer_framework judge

重写 `defer_framework` 评测:把原来的「stage1 整体判 solved → stage2 逐步判相关 → stage3 整体判 truthful」三段式,改成 **一个代码驱动、维护评测记忆的逐步 agent judge**。

## 动机

原三段式把 `answer_points` 同时用于两类性质不同的判断,但不区分它们,也不跟踪「每个要点到底在哪一步、被什么证据满足」:

- **answer-bearing 要点**:最终答案文本里必须直接出现的内容(如"必须给出具体食谱名"、"必须列出配料")。
- **evidence-bearing 要点**:无法只看答案文本确认、必须从轨迹证据验证的内容(如"评分≥4 星"、"评论>25 条"、"这是排序后的第二名")。

stateful judge 显式区分这两类,并维护一份逐步演化的**要点状态表(评测记忆)**:对每个要点记录「截至当前做过什么、是否已判定完成、是哪一步的什么证据满足的」。

## 数据结构:要点状态表(评测记忆)

```jsonc
{
  "point_id": 0,
  "text": "The recipe must have a rating of at least 4 stars.",
  "kind": "evidence_bearing",        // 或 "answer_bearing";分类阶段产出
  "status": "pending",               // pending | satisfied | contradicted | partial
  "satisfied_at_step": null,         // 首次判 satisfied 的 step 号
  "supporting_step_ids": [],         // 累积的支撑步
  "evidence_log": [                  // 评测记忆:每次状态变化追加一条
    // {"step": 5, "status": "satisfied", "note": "页面显示 4.6 stars"}
  ],
  "last_note": ""                    // 最近一次判断摘要,喂给下一步省 token
}
```

整张表 = `list[PointState]`。它在循环中被原地更新,既是 judge 的工作记忆,也是最终落盘的可解释轨迹。

## 流程(代码驱动)

```
evaluate_defer_framework_stateful(task, annotation, ...):
  points = get_defer_answer_key_points()           # 复用现有取点逻辑
  if not prediction: return label 0

  # 阶段 A:分类(1 次 LLM)——把每个 point 分成 answer_bearing / evidence_bearing
  kinds = classify_points(task, points)            # prompts/defer_classify_points.txt
  state = init_state(points, kinds)                # 全 pending

  # 阶段 B:预压缩(0 LLM)——折叠连续重复的导航步,只保留状态可能变化的步
  steps = compress_navigation(get_pre_answer_records(records))

  # 阶段 C:逐步更新评测记忆(每步 1 次 LLM)
  for step in steps:
      state = update_state(task, state, step)       # prompts/defer_update_state.txt
      # LLM 只允许:把 evidence_bearing 要点 pending→satisfied/contradicted/partial,
      #            追加 evidence_log + supporting_step_ids;不得发明新要点。

  # 阶段 D:最终结算(1 次 LLM)——读最终答案,判 answer_bearing 要点 + 整体 label
  verdict = finalize(task, state, prediction)       # prompts/defer_finalize.txt
  return label + 完整 state(评测记忆)落盘
```

LLM 调用数 ≈ `1(分类) + 压缩后步数 + 1(结算)`。对比原三段式 stage2 本就逐步(每 pre-answer step 1 次),数量级不变;压缩后通常更少。

`--defer-temperature` 默认 0(可复现);要多样本投票时调高 + `--repeat-count`。

## 预压缩规则(纯代码、不改语义)

连续步若满足以下全部,折叠为一条(保留首步,丢弃后续重复):
- 同一 `tool` 且同一 `current_url`;
- `observation` / `som_text` 与上一保留步实质相同(归一化后相等);
- 不是 answer 步。

折叠记录数写进结果(`collapsed_steps`),保证可追溯——不静默丢步。

## 输出兼容

为不打断既有下游(summary 按 `predicted_label` / `evaluation_status` 统计,可视化读 stage 字段),新结果:
- 保留 `evaluation_type="defer_framework"`、`predicted_label`、`evaluation_status`、`reason`、`annotation`/`classification` 等顶层字段;
- 新增 `stateful=true`、`point_states`(最终状态表)、`step_updates`(逐步状态快照,供可视化逐步回放)、`classification_of_points`、`collapsed_steps`;
- 兼容性映射:同时写出 `stage1.solved`(= 所有 answer_bearing 满足)、`stage2.selected_step_ids`(= 所有 supporting_step_ids 并集)、`stage3.truthful`(= 最终 label),让旧的读 stage 的脚本不炸。

## 判定语义

- `predicted_label=1` 当且仅当:所有 answer_bearing 要点在最终答案中满足 **且** 所有 evidence_bearing 要点状态为 satisfied(无 contradicted)。
- 任一要点 contradicted → label 0。
- partial 视作未满足(label 0),但在 reason 里点名是哪条、缺什么。

## Phase A 预处理与缓存

Phase A(把每个 answer key point 分成 `answer_bearing` / `evidence_bearing`)只依赖 `(task_id, question, answer_key_points)`,与具体 run / repeat / 评测模式无关,因此可以一次性预算好、跨多次评测复用。

- 预处理脚本:`scripts/classify_defer_answer_points.py`,读 `annotations.json` 中 `annotation_types` 含 `defer_framework` 且 `answer_key_points` 非空的任务,逐任务调一次 LLM,把分类结果落地为 JSONL。
- 文件命名约定:`outputs/<dataset_stem>.point_classification.<model_slug>.jsonl`,例如 `outputs/web_insta_v3_test50_nohuman_seed42.point_classification.kimi-k2.5.jsonl`。模型身份编码在文件名里,不同模型走不同文件,文件本身不再按 model 加键。
- 评测器接入:`scripts/evaluate_classified_web_tasks.py` 新增 `--point-classification <path>`。命中且 `answer_key_points` 长度+逐条文本均匹配时直接复用缓存的 `kinds`,跳过 Phase A 的 LLM 调用;未命中或不匹配则回落到原有 live 路径。命中与否通过结果行 `classification_of_points.source = "cache" | "live"` 记录,便于后续审计。
