"""discard-all(HYBRID_DISCARD_ALL=1):触线原地清空、轨迹完整落盘、不折叠、不超线。

env 的设置/还原必须包在函数里:ReactHistory 在 __init__ 读 env,而 pytest 把整个
目录跑在同一个进程里 —— 在模块顶层 export 会污染同批的其他用例(实测把
test_react_v2_parser 和 post_action_screenshot 一起带挂)。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import react_v2 as rv  # noqa: E402

ENV = {
    "HYBRID_DISCARD_ALL": "1",
    "HYBRID_FOLD_HIGH": "3000",     # 人为压低触发线
    "HYBRID_FOLD_LOW": "1500",
    "HYBRID_DISCARD_MIN_GAP": "3",
}


class DiscardAllTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in list(ENV) + ["HYBRID_DISCARD_KEEP_LAST"]}
        os.environ.update(ENV)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _run(self, n_steps=40):
        h = rv.ReactHistory(max_input_tokens=8000)
        q = "原始问题:某年某人拿了什么奖?"
        h.add_user_task(q)
        big = "观测内容 " * 400
        for i in range(1, n_steps + 1):
            h.add_step(side="S",
                       action_text='{"name":"search","arguments":{"query":"q%d"}}' % i,
                       observation=f"[step {i}] " + big, think=f"想法{i}")
            msgs = h.render_multiturn_capped("SYS", "S", q)
            self.assertLessEqual(h._tokens_of(msgs), 3000, f"第{i}步渲染后仍超线")
        return h, q

    def test_discards_in_place_and_keeps_trajectory(self):
        h, q = self._run()
        self.assertGreater(len(h.discard_events), 0, "应当触发过丢弃")
        self.assertEqual(len(h.folded), 0, "discard-all 模式下不该有折叠")
        self.assertEqual(len([t for t in h.turns if t.get("role") == "step"]), 40,
                         "轨迹必须完整保留,只是渲染时跳过")
        txt = str(h.render_multiturn_capped("SYS", "S", q))
        self.assertIn("原始问题", txt, "原题必须留着")
        self.assertNotIn("[step 1]", txt, "丢弃点之前的步骤不该进上下文")
        st = h.stats()
        self.assertEqual(st["discard_all_count"], len(h.discard_events))

    def test_keep_last_retains_recent_steps(self):
        os.environ["HYBRID_DISCARD_KEEP_LAST"] = "2"
        h, q = self._run(20)
        ev = h.discard_events[0]
        self.assertEqual(ev["kept_last"], 2)
        # 丢弃刚发生的那一轮,可见步骤应当 == keep_last
        self.assertGreater(len(h.discard_events), 0)

    def test_off_by_default(self):
        os.environ.pop("HYBRID_DISCARD_ALL", None)
        h = rv.ReactHistory(max_input_tokens=8000)
        self.assertFalse(h.discard_all_enabled)
        self.assertEqual(h.discard_start, 0)


if __name__ == "__main__":
    unittest.main()


class SummaryScopeTest(DiscardAllTest):
    """后验渲染必须看到**整条**轨迹,且不能污染 live 历史。"""

    def test_full_render_sees_discarded_steps_and_restores_state(self):
        h, q = self._run(40)
        self.assertGreater(len(h.discard_events), 0)
        before = (h.discard_start, h.seg_start_override, set(h.folded), h._seg_shot)

        visible = str(h.render_multiturn_capped("SYS", "S", q))
        self.assertNotIn("[step 1]", visible, "可见段不该有被丢弃的步")

        full = str(h.render_full_for_summary("SYS", "S", q))
        self.assertIn("step 1", full, "全量渲染必须看得到被丢弃的步")
        self.assertGreater(len(full), len(visible), "全量渲染应当比可见段长得多")

        after = (h.discard_start, h.seg_start_override, set(h.folded), h._seg_shot)
        self.assertEqual(before[0], after[0], "discard_start 必须还原")
        self.assertEqual(before[1], after[1], "seg_start_override 必须还原")
        self.assertEqual(before[2], after[2], "folded 必须还原(全量渲染会折叠,不能污染 live 历史)")
        # 还原后可见段必须和渲染前逐字相同
        self.assertEqual(visible, str(h.render_multiturn_capped("SYS", "S", q)))
