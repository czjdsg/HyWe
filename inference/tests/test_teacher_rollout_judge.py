import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from collector.judge_teacher_rollouts import (
    SCHEMA_VERSION,
    actual_route,
    cleanup_task_images,
    extract_judge_json,
    reference_ids,
    reference_component_specs,
    render_text,
    selected_image_events,
    validate_judgment,
)


def sample_task():
    return {
        "task_id": "task_1",
        "question": "Find a public fact, then use it in a live form.",
        "private_teacher_steps": {
            "switch_form": "F3",
            "difficulty": "easy",
            "start_mode": "S",
            "expected_mode_route": ["S", "O"],
            "expected_switch_count": 1,
            "steps": [
                {
                    "type": "phase",
                    "phase_id": "p1",
                    "mode": "S",
                    "components": [
                        {
                            "component_id": "s1",
                            "goal": "Resolve the fact",
                            "done_when": "The fact is verified",
                            "produces": "s1.result",
                            "consumes": [],
                        }
                    ],
                },
                {
                    "type": "switch",
                    "policy_id": "sw1",
                    "from_mode": "S",
                    "to_mode": "O",
                    "trigger": "The fact is ready",
                },
                {
                    "type": "phase",
                    "phase_id": "p2",
                    "mode": "O",
                    "components": [
                        {
                            "component_id": "o1",
                            "goal": "Use the fact",
                            "done_when": "The form visibly accepts it",
                            "produces": "o1.result",
                            "consumes": ["s1.result"],
                        }
                    ],
                },
            ],
            "dependency_edges": [
                {"from": "s1.result", "to": "o1", "binding_role": "form input"}
            ],
            "answer_schema": {"result": "string"},
        },
    }


def sample_result(first_image: str = ""):
    return {
        "termination": "answer",
        "prediction": "done",
        "records": [
            {
                "step": 1,
                "active_mode_before": "S",
                "active_mode_after": "O",
                "raw_response": (
                    "<think>Fact verified.</think>"
                    '<tool_call>{"name":"switch","arguments":{"reason":"ready"}}</tool_call>'
                ),
                "tool_call": {"name": "switch", "arguments": {"reason": "ready"}},
                "observation": "[switch] now O",
                "img_path": first_image,
            },
            {
                "step": 2,
                "active_mode_before": "O",
                "active_mode_after": "O",
                "current_url": "https://example.com/form",
                "raw_response": (
                    "<think>Use the form.</think>"
                    '<tool_call>{"name":"click","arguments":{"x":500,"y":500}}</tool_call>'
                ),
                "tool_call": {"name": "click", "arguments": {"x": 500, "y": 500}},
                "observation": "success",
                "img_path": first_image,
            },
            {
                "step": 3,
                "active_mode_before": "O",
                "active_mode_after": "O",
                "raw_response": "<think>Complete.</think><answer>done</answer>",
                "tool_call": None,
                "observation": "",
                "img_path": first_image,
            },
        ],
    }


class TeacherRolloutJudgeTest(unittest.TestCase):
    def test_reference_and_route_are_rendered_without_repeated_system_prompts(self):
        rendered, metadata = render_text(sample_task(), sample_result())
        self.assertIn("<private_expected_solution_contract>", rendered)
        self.assertIn("<trajectory_step id=\"1\"", rendered)
        self.assertIn("<trajectory_step id=\"3\"", rendered)
        self.assertNotIn("system_prompt", rendered)
        self.assertEqual(metadata["component_ids"], ["s1", "o1"])
        self.assertEqual(metadata["component_specs"], [("s1", "S"), ("o1", "O")])
        self.assertEqual(metadata["policy_ids"], ["sw1"])
        self.assertEqual(metadata["actual_route"], ["S", "O"])
        self.assertTrue(metadata["exact_route_match"])

    def test_actual_route_collects_switch_steps(self):
        route, steps = actual_route(sample_result()["records"])
        self.assertEqual(route, ["S", "O"])
        self.assertEqual(steps, [1])

    def test_rejected_switch_is_not_counted_as_an_executed_route_change(self):
        result = sample_result()
        result["records"][0]["active_mode_after"] = "S"
        result["records"][0]["switch_executed"] = False
        route, steps = actual_route(result["records"])
        self.assertEqual(route, ["S", "O"])
        self.assertEqual(steps, [])

    def test_feedback_windows_render_feedback_and_affected_teacher_step(self):
        phase_control = {
            "events": [
                {
                    "event_id": 4,
                    "trigger": "periodic",
                    "trajectory_step": 1,
                    "feedback": "Inspect the primary source before switching.",
                    "feedback_delivered_to_step": 2,
                }
            ]
        }
        rendered, metadata = render_text(sample_task(), sample_result(), phase_control)
        self.assertIn("Inspect the primary source before switching", rendered)
        self.assertIn("Use the form", rendered)
        self.assertEqual(metadata["controller_feedback_event_ids"], [4])

    def test_image_selection_uses_o_entry_and_post_action_image(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            trajectory = task_dir / "trajectory"
            trajectory.mkdir()
            entry = trajectory / "switch_1_full_screenshot.png"
            post = trajectory / "post_2_full_screenshot.png"
            entry.write_bytes(b"entry")
            post.write_bytes(b"post")
            result = sample_result(str(entry))
            events = selected_image_events(task_dir, result)
            self.assertEqual(
                [(event["step"], event["placement"]) for event in events],
                [(1, "after_switch_to_o"), (2, "after")],
            )

    def test_extract_and_validate_judgment(self):
        components, policies = reference_ids(
            json.loads(
                render_text(sample_task(), sample_result())[0].split(
                    "<private_expected_solution_contract>\n", 1
                )[1].split("\n</private_expected_solution_contract>", 1)[0]
            )
        )
        judgment = {
            "schema_version": SCHEMA_VERSION,
            "task_id": "task_1",
            "trajectory_success": True,
            "success": {
                "verdict": "success",
                "component_checks": [
                    {
                        "component_id": "s1",
                        "expected_mode": "S",
                        "status": "satisfied",
                        "evidence_steps": [1],
                    },
                    {
                        "component_id": "o1",
                        "expected_mode": "O",
                        "status": "satisfied",
                        "evidence_steps": [2],
                    },
                ],
                "switch_checks": [{"policy_id": "sw1"}],
            },
            "cot_leakage": True,
            "leakage": {
                "verdict": "leak",
                "first_leak_step": 1,
                "leak_steps": [
                    {
                        "step": 1,
                        "exact_quote": "private teacher solution",
                    }
                ],
            },
            "controller_feedback_audit": {
                "verdict": "no_leak",
                "events": [],
            },
            "audited_agent_steps": 3,
            "audited_controller_feedback_events": 0,
        }
        raw = "<think>audit</think><judge_result>" + json.dumps(judgment) + "</judge_result>"
        parsed = extract_judge_json(raw)
        validate_judgment(
            parsed,
            "task_1",
            3,
            components,
            policies,
            reference_component_specs(sample_task()["private_teacher_steps"]),
            {1: "S", 2: "O", 3: "O"},
        )


if __name__ == "__main__":
    unittest.main()
