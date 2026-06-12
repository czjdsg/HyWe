# Stateful operation_key_points judge

Type 4(页面状态验证)与 stateful defer 同构,改用同一套「要点状态表逐步更新」引擎,只多一个 **timing** 维度。本文档同时记录抽出的公共引擎。

## 为什么同构

- defer:维护 answer_points 的完成状态,逐步读轨迹更新 → 结算。
- operation:维护 page_states 的达成状态,逐步读轨迹更新 → 结算。

唯一差别:operation 的要点带 `timing`(intermediate / final),defer 的要点带 `kind`(answer_bearing / evidence_bearing)。两者都是「要点的一个类别标签,决定它何时/如何被判定满足」。所以抽象成同一个 **point-kind 策略**。

## 公共引擎(kind-agnostic)

抽出的、defer 和 operation 共用的部分(已在 `evaluate_classified_web_tasks.py`):

- `init_point_states(points, kinds)` —— 建状态表,每点 `{point_id, text, kind, status, satisfied_at_step, supporting_step_ids, evidence_log, last_note}`。
- `compress_navigation(records)` —— 两层无损压缩(连续相同 + 全局已见 browser 签名去重)。
- `render_memory_block(states)` —— 把状态表渲染成喂给 LLM 的文本。
- `walk_trajectory(task, states, steps, notes, system_prompt, defer_kinds, ...)` —— 通用逐步循环:对每个压缩后 step,喂「当前状态表 + 这一步」给 `system_prompt`,把返回的逐点更新 merge 进状态表。`defer_kinds` 是「本阶段不更新、留到 finalize 的 kind 集合」(defer 传 `{"answer_bearing"}`,operation 传 `set()` —— 所有 page_state 都在 walk 中更新)。

merge 规则(对所有 kind 通用):
- 只有这一步带 `new_evidence` 才改状态、记 `evidence_log`、加 `supporting_step_ids`;
- 不把 `satisfied` 静默降级;
- `kind ∈ defer_kinds` 的点在 walk 阶段只记笔记、保持 pending。

## operation 的 timing 语义(本次决策)

- **intermediate**:轨迹中任一步达成即永久 `satisfied`,后续离开该页不影响。walk 阶段判定;一旦 satisfied 不回退。
- **final**:要求在轨迹末尾 / 最终浏览状态仍然成立。walk 阶段可先标 `satisfied`(到过该终态页),但 **finalize 阶段对所有 final 要点做一次末态复核** —— 如果最后又离开了该终态(末态不再呈现),降级为 `partial`/`contradicted`。

> 不加跨要点的顺序约束(intermediate 必须在 final 之前)——标注未必有此意图,避免误杀。

## operation 流程

```
evaluate_operation_key_points_stateful(task, annotation, ...):
  page_states = annotation.page_states            # [{timing, state}, ...]
  if not page_states: return missing
  points = [s["state"] for s in page_states]
  kinds  = [s["timing"] or "final" for s in page_states]   # timing 即 kind
  states = init_point_states(points, kinds)

  steps = compress_navigation(task["records"])     # 注意:含 answer 步也无妨,operation 看页面态
  walk_trajectory(task, states, steps, notes,
                  system_prompt=prompts["operation_update_state"],
                  defer_kinds=set())                # 全部在 walk 中更新

  finalize_operation(task, states, prompts["operation_finalize"])
    # 只复核 final 要点的末态;intermediate 保持 walk 结论
  predicted_label = 1 iff 所有要点 status==satisfied
```

LLM 调用 ≈ 压缩后步数 + 1(finalize)。对比原实现 N 个 page_state × 全轨迹各一次,长轨迹下新方案通常更省且能感知时序。

## 输出兼容

保留 `evaluation_type="operation_key_points"`、`predicted_label`、`evaluation_status`、`reason`;新增 `stateful=true`、`page_state_results`(= 状态表,评测记忆)、`step_updates`、`collapsed_steps`。同时投影旧的 `key_point_results`(每条含 `key_point/page_state/timing/achieved/supporting_step_ids/reason`)让既有读者不炸。

## 组合类型 [3,4] / [1,4]

`combine_evaluation_results` 不变(AND 合并):defer/static 子结果 + operation 子结果都为 1 才整体 1。两条子路由各自 stateful,互不影响。
