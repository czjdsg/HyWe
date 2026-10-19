import unittest
from types import SimpleNamespace

from hybrid_agent import HybridBrowserAgent


def agent_for(model: str, coord_scale: int):
    agent = object.__new__(HybridBrowserAgent)
    agent.args = SimpleNamespace(
        model=model,
        routed_operation_model=model,
        coord_scale=coord_scale,
        window_width=1280,
        window_height=720,
    )
    return agent


class KimiCoordinateAdapterTest(unittest.TestCase):
    def test_kimi_accepts_relative_coordinates(self):
        agent = agent_for("kimi-k2.6", 1)
        args = {"x": 0.5, "y": 0.25}
        scale = agent._effective_coord_scale(args)
        self.assertEqual(scale, 1)
        self.assertEqual(
            agent._scale_xy(args["x"], args["y"], scale=scale),
            (640.0, 180.0),
        )

    def test_kimi_accepts_1000_grid_coordinates(self):
        agent = agent_for("kimi-k2.6", 1)
        args = {"x": 500, "y": 250}
        scale = agent._effective_coord_scale(args)
        self.assertEqual(scale, 1000)
        self.assertEqual(
            agent._scale_xy(args["x"], args["y"], scale=scale),
            (640.0, 180.0),
        )

    def test_drag_uses_one_scale_for_both_endpoints(self):
        agent = agent_for("kimi-k2.6", 1)
        args = {
            "from_x": 0.5,
            "from_y": 0.5,
            "to_x": 800,
            "to_y": 900,
        }
        scale = agent._effective_coord_scale(args)
        self.assertEqual(scale, 1000)
        self.assertEqual(
            agent._scale_xy(
                args["from_x"],
                args["from_y"],
                scale=scale,
            ),
            (0.64, 0.36),
        )
        self.assertEqual(
            agent._scale_xy(args["to_x"], args["to_y"], scale=scale),
            (1024.0, 648.0),
        )

    def test_logical_edge_is_clamped_to_last_viewport_pixel(self):
        agent = agent_for("kimi-k2.6", 1)
        self.assertEqual(
            agent._scale_xy(1, 1, scale=1),
            (1279.0, 719.0),
        )
        self.assertEqual(
            agent._scale_xy(1000, 1000, scale=1000),
            (1279.0, 719.0),
        )

    def test_qwen_is_always_1000_grid(self):
        agent = agent_for("qwen3.8-max", 1000)
        self.assertEqual(
            agent._effective_coord_scale({"x": 0.5, "y": 0.25}),
            1000,
        )
        self.assertEqual(
            agent._effective_coord_scale({"x": 500, "y": 250}),
            1000,
        )

    def test_unknown_model_does_not_get_kimi_auto_detection(self):
        agent = agent_for("other-vlm", 1)
        self.assertEqual(
            agent._effective_coord_scale({"x": 500, "y": 250}),
            1,
        )


if __name__ == "__main__":
    unittest.main()
