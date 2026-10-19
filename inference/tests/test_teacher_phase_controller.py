import unittest

from react_v2 import ReactHistory
from teacher_phase_controller import (
    TeacherPhaseController,
    phase_termination_reason,
    sanitize_teacher_feedback,
)


def contract():
    return {
        "start_mode": "S",
        "expected_mode_route": ["S", "O"],
        "expected_switch_count": 1,
        "answer_schema": ["final result"],
        "steps": [
            {
                "type": "phase",
                "phase_id": "p1",
                "mode": "S",
                "components": [
                    {
                        "component_id": "s1",
                        "goal": "Find the public value needed for the form.",
                        "done_when": "The exact value is confirmed by a primary source.",
                        "produces": "s1.result",
                        "consumes": [],
                        "sites": [{
                            "domain": "source.example",
                            "entry_url": "https://source.example/value",
                        }],
                    }
                ],
            },
            {
                "type": "switch",
                "policy_id": "sw1",
                "from_mode": "S",
                "to_mode": "O",
                "trigger": "The exact source value has been found.",
            },
            {
                "type": "phase",
                "phase_id": "p2",
                "mode": "O",
                "components": [
                    {
                        "component_id": "o1",
                        "goal": "Enter the verified value in the live form and submit it.",
                        "done_when": "The success state is visibly rendered.",
                        "produces": "o1.result",
                        "consumes": ["s1.result"],
                        "sites": [{
                            "domain": "form.example",
                            "entry_url": "https://form.example/submit",
                        }],
                    }
                ],
            },
        ],
    }


class TeacherPhaseControllerTest(unittest.TestCase):
    def test_card_exposes_only_current_goal_and_public_sites(self):
        controller = TeacherPhaseController(contract(), "S")
        card = controller.render_teacher_context()
        self.assertIn("Find the public value needed for the form", card)
        self.assertIn("source.example", card)
        self.assertNotIn("Enter the verified value", card)
        self.assertNotIn("form.example", card)
        self.assertNotIn("done_when", card)
        self.assertNotIn("component_id", card)
        self.assertNotIn("expected_mode_route", card)
        self.assertNotIn("p1", card)

    def test_runtime_start_mode_must_match_private_contract(self):
        with self.assertRaises(ValueError):
            TeacherPhaseController(contract(), "O")

    def test_verified_switch_advances_once_and_preserves_binding_evidence(self):
        controller = TeacherPhaseController(contract(), "S")
        controller.apply_phase_verdict({
            "phase_complete": True,
            "evidence_steps": [2, 3],
            "produced_bindings": {"s1.result": "VALUE-42"},
        })
        self.assertTrue(controller.decide_switch().allow)
        controller.advance_after_switch("O", next_step=5)
        self.assertEqual(controller.current_phase_index, 2)
        self.assertEqual(controller.current_phase["mode"], "O")
        self.assertEqual(controller.bindings["s1.result"]["evidence_steps"], [2, 3])
        card = controller.render_teacher_context()
        self.assertIn("VALUE-42", card)
        self.assertIn("trajectory step(s) 2, 3", card)

    def test_premature_answer_and_switch_are_rejected(self):
        controller = TeacherPhaseController(contract(), "S")
        self.assertFalse(controller.decide_switch().allow)
        self.assertFalse(controller.decide_answer().allow)
        self.assertEqual(controller.current_phase_index, 1)

    def test_final_answer_requires_final_phase_verification(self):
        controller = TeacherPhaseController(contract(), "S")
        controller.apply_phase_verdict({
            "phase_complete": True,
            "evidence_steps": [1],
            "produced_bindings": {"s1.result": "x"},
        })
        controller.advance_after_switch("O", next_step=3)
        self.assertFalse(controller.decide_answer().allow)
        controller.apply_phase_verdict({
            "phase_complete": True,
            "evidence_steps": [4],
            "produced_bindings": {"o1.result": "submitted"},
        })
        self.assertTrue(controller.decide_answer().allow)

    def test_leading_switch_is_current_action_not_future_phase_leak(self):
        value = contract()
        value["start_mode"] = "O"
        value["steps"].insert(0, {
            "type": "switch",
            "policy_id": "leading",
            "from_mode": "O",
            "to_mode": "S",
            "trigger": "Use indexed public content.",
        })
        controller = TeacherPhaseController(value, "O")
        card = controller.render_teacher_context()
        self.assertIn("search-based", card)
        self.assertNotIn("Find the public value", card)
        self.assertTrue(controller.decide_switch().allow)
        controller.advance_after_switch("S", next_step=2)
        self.assertEqual(controller.current_phase["mode"], "S")

    def test_feedback_is_linked_to_the_teacher_step_that_received_it(self):
        controller = TeacherPhaseController(contract(), "S")
        event_id = controller.record_verifier_event(
            trigger="periodic",
            trajectory_step=8,
            verdict={"phase_complete": False},
            raw_response="private raw",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            feedback="Inspect a primary source before switching.",
        )
        self.assertEqual(controller.pending_feedback_ids(), [event_id])
        self.assertEqual(controller.mark_feedback_delivered(9), [event_id])
        self.assertEqual(controller.events[0]["feedback_delivered_to_step"], 9)

    def test_feedback_sanitizer_removes_schema_labels(self):
        value = sanitize_teacher_feedback(
            "component_id s1 has not met done_when in phase p1",
            "continue",
        )
        self.assertNotIn("component_id", value)
        self.assertNotIn("done_when", value)
        self.assertNotIn("p1", value)

    def test_rejected_switch_does_not_start_a_new_history_segment(self):
        history = ReactHistory(max_o_screenshots=-1)
        history.add_step("S", '{"name":"search","arguments":{"query":["x"]}}', "result")
        history.add_step(
            "S",
            '{"name":"switch","arguments":{"reason":"too early"}}',
            "continue searching",
            switch_executed=False,
        )
        self.assertEqual(history.segment_start(), 0)

    def test_verifier_evidence_restores_and_protects_source_observation(self):
        history = ReactHistory(max_o_screenshots=-1)
        history.add_step(
            "S",
            '{"name":"visit","arguments":{"url":"https://example.test","goal":"value"}}',
            "the complete primary-source tool result",
        )
        turn = history.turns[0]
        turn["text"] = "folded"
        history.folded.add(id(turn))
        history.protect_steps([1])
        self.assertEqual(turn["text"], "the complete primary-source tool result")
        self.assertTrue(turn["protect_observation"])
        self.assertNotIn(id(turn), history.folded)

    def test_three_explicit_failures_trigger_no_progress_check(self):
        controller = TeacherPhaseController(contract(), "S")
        for step in range(1, 4):
            controller.note_action(
                step,
                {"name": "visit", "arguments": {"url": f"https://example/{step}"}},
                "Action execution result:\n- status: failure",
            )
        self.assertEqual(
            controller.should_periodic_check(8, 40, no_progress_actions=3),
            (True, "no_progress"),
        )

    def test_deep_search_gets_100_actions_while_direct_search_and_o_get_50(self):
        value = contract()
        value["steps"][0]["components"][0]["depth"] = "deep"
        controller = TeacherPhaseController(value, "S")

        self.assertTrue(controller.current_phase_is_deep_search)
        self.assertEqual(controller.phase_action_budget(50, 100), 100)
        for step in range(1, 51):
            controller.note_action(step, {"name": "search"}, f"result-{step}")
        self.assertEqual(
            controller.should_periodic_check(
                0,
                50,
                deep_search_max_actions=100,
            ),
            (False, ""),
        )
        for step in range(51, 101):
            controller.note_action(step, {"name": "search"}, f"result-{step}")
        self.assertEqual(
            controller.should_periodic_check(
                0,
                50,
                deep_search_max_actions=100,
            ),
            (True, "phase_budget"),
        )

        controller.apply_phase_verdict({"phase_complete": True})
        controller.advance_after_switch("O", next_step=102)
        self.assertFalse(controller.current_phase_is_deep_search)
        self.assertEqual(controller.phase_action_budget(50, 100), 50)

        direct_value = contract()
        direct_value["steps"][0]["components"][0]["depth"] = "shallow"
        direct_controller = TeacherPhaseController(direct_value, "S")
        self.assertFalse(direct_controller.current_phase_is_deep_search)
        self.assertEqual(direct_controller.phase_action_budget(50, 100), 50)

    def test_recoverable_verifier_block_is_returned_as_feedback(self):
        verdict = {
            "phase_complete": False,
            "blocked": True,
            "block_kind": "site_error",
        }
        self.assertEqual(phase_termination_reason("periodic", verdict), "")
        self.assertEqual(phase_termination_reason("no_progress", verdict), "")

    def test_captcha_and_exhausted_phase_budget_are_hard_stops(self):
        self.assertEqual(
            phase_termination_reason(
                "periodic",
                {"phase_complete": False, "blocked": True, "block_kind": "captcha"},
            ),
            "captcha",
        )
        self.assertEqual(
            phase_termination_reason(
                "periodic",
                {"phase_complete": False, "blocked": False, "block_kind": "none"},
                captcha_seen=True,
            ),
            "captcha",
        )
        self.assertEqual(
            phase_termination_reason(
                "phase_budget",
                {"phase_complete": False, "blocked": True, "block_kind": "site_error"},
            ),
            "phase_budget_exceeded",
        )
        self.assertEqual(
            phase_termination_reason(
                "phase_budget",
                {"phase_complete": True, "blocked": False, "block_kind": "none"},
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
