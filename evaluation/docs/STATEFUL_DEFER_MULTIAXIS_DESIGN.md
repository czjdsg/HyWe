# 多维度要点判定 (dual-axis points)

## 问题

一条 answer key point 可能**同时**要求两件事:
- 答案文本里要直接陈述它(answer 维度)
- 且轨迹/页面要真实证实它(evidence 维度)

例:`"The answer must verify that the selected business has a customer rating strictly above 4.0"`
—— 评分 >4.0 既要在答案里报告(answer),又要在源页面上成立(evidence)。

原设计强制每条要点二选一(选更强的 evidence),丢掉了 answer 维度:finalize 不会去查答案文本是否真的报告了评分。约 31% 的现 evidence 单标要点疑似属于这种双重要点。

## 方案:每条要点带 1-2 个独立 axis 子判定

point_state 结构(向后兼容地扩展,不破坏现有字段):

```jsonc
{
  "point_id": 0,
  "text": "...rating strictly above 4.0",
  "kind": "evidence_bearing",     // 保留: 单值,取"主导维度"(dual 时 = evidence)。
                                   // operation 路由 + stage1/stage3 投影继续读它,零改动。
  "axes": [                        // 新增: 1 或 2 个独立维度
    {"axis": "evidence", "status": "pending", "satisfied_at_step": null,
     "supporting_step_ids": [], "evidence_log": [], "last_note": ""},
    {"axis": "answer",   "status": "pending", "satisfied_at_step": null,
     "supporting_step_ids": [], "evidence_log": [], "last_note": ""}
  ],
  // 下面是聚合字段,从 axes 派生,保持旧读者可用:
  "status": "pending",             // = 所有 axis 都 satisfied 才 satisfied;任一 contradicted 则 contradicted
  "satisfied_at_step": null,       // = 最后一个变 satisfied 的 axis 的步号
  "supporting_step_ids": [],       // = 所有 axis 的并集
  "evidence_log": [],              // = 所有 axis 的合并(带 axis 标注)
  "last_note": ""
}
```

### kind 与 axes 的关系
- 单 axis evidence → `kind="evidence_bearing"`, `axes=[evidence]`
- 单 axis answer → `kind="answer_bearing"`, `axes=[answer]`
- dual → `kind="evidence_bearing"`(主导取 evidence,因为它更强), `axes=[evidence, answer]`

`kind` 永远是单值,operation 路由(它把 timing 塞 kind)和 stage 投影完全不受影响。

## walk 阶段(逐步更新)

只推进 `axis == "evidence"` 的子判定(answer axis 留给 finalize)。
- `deferred_kinds` 语义改为按 **axis** 过滤:walk 跳过 answer axis,更新 evidence axis。
- `apply_step_update` 把 LLM 返回的逐点状态应用到该点的 evidence axis 上(单 axis answer 点在 walk 中完全不动)。
- 每个 axis 维护自己的 status/evidence_log/supporting_step_ids;然后重算聚合字段。

## finalize 阶段

- 对每条要点的 `axis == "answer"` 子判定,按最终答案文本核验。
- evidence axis 保留 walk 的结论(finalize 仍可做一致性检查:若答案承诺的实体与 evidence 支持的不一致 → 把 evidence axis 标 contradicted)。
- 重算聚合 status。

## label 派生

```
point.status = satisfied  iff  所有 axis.status == satisfied
point.status = contradicted iff 任一 axis.status == contradicted
否则 = partial 或 pending
predicted_label = 1 iff 所有 point.status == satisfied
```

等价于:把每条要点的所有维度都摊平,要求全部 satisfied。

## 向后兼容(全部保持)

- `kind`(单值)不变 → operation 路由、`stage1.solved`(所有 answer 维度满足)、`stage3.truthful`(所有 evidence 维度满足)、`stage2.selected_step_ids` 全部照常。
- 聚合的 `status/supporting_step_ids/evidence_log/satisfied_at_step/last_note` 仍在顶层 → 旧读者无感。
- 新增 `axes` 字段是纯增量,旧消费者忽略它即可。
- 单 axis 要点的行为与现在**逐字节一致**(axes 只有一个元素,聚合 == 该元素)。

## 缓存 schema 升级

`classifications[i]` 从:
```jsonc
{"point_id":0, "text":"...", "kind":"evidence_bearing", "reason":"..."}
```
变为:
```jsonc
{"point_id":0, "text":"...", "axes":["evidence","answer"], "kind":"evidence_bearing", "reason":"..."}
```
- 新增 `axes`(1-2 个,有序:evidence 在前如果存在)。
- `kind` 仍写(= axes 里的主导维度),让旧加载器和 length 校验不变。
- `load_point_classification` 返回里增加 `axes_by_point`;evaluate_defer_framework 用 axes 构造 point_states,缺 axes 时从 kind 退化为单 axis(完全向后兼容旧缓存)。

## operation 路由

完全不碰。它走 `evaluate_operation_key_points`,用 `kind` 存 timing(intermediate/final),不经过 axes 逻辑。多 axis 是 defer 专属。
