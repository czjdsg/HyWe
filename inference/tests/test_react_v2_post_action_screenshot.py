import base64
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hybrid_agent import HybridBrowserAgent
from PIL import Image
from react_v2 import ReactHistory


class _HistorySpy:
    def __init__(self):
        self.calls = []

    def add_step(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _WebStub:
    def __init__(self, post_path):
        self.post_path = post_path
        self.calls = []

    async def current_state(self, step, prefix=""):
        self.calls.append((step, prefix))
        return {
            "img_path": str(self.post_path),
            "current_url": "https://example.test/after",
            "current_title": "After action",
        }


class ReactV2PostActionScreenshotTest(unittest.IsolatedAsyncioTestCase):
    async def test_v2_forced_final_accepts_plain_gateway_answer(self):
        agent = HybridBrowserAgent.__new__(HybridBrowserAgent)

        parsed, error = agent._v2_parse(
            "The best-supported answer is Example Insurance.",
            is_final=True,
        )

        self.assertEqual(error, "")
        self.assertEqual(
            parsed["answer"],
            "The best-supported answer is Example Insurance.",
        )
        self.assertIsNone(parsed["tool_call"])

    async def test_v2_o_step_attaches_post_action_screenshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pre_path = Path(tmp_dir) / "pre.png"
            post_path = Path(tmp_dir) / "post.png"
            pre_path.write_bytes(b"PRE")
            post_path.write_bytes(b"POST")

            agent = HybridBrowserAgent.__new__(HybridBrowserAgent)
            agent.web = _WebStub(post_path)
            agent._hist = _HistorySpy()
            record = {
                "step": 7,
                "active_mode_before": "O",
                "img_path": str(pre_path),
                "tool_call": {
                    "name": "click",
                    "arguments": {"x": 10, "y": 20},
                },
                "observation": "clicked",
                "report_parsed": {"think": "click the visible button"},
            }

            await agent._v2_add_executed_step_to_history(record)

            self.assertEqual(agent.web.calls, [(7, "post_")])
            self.assertEqual(record["post_shot_path"], str(post_path))
            self.assertEqual(
                record["post_current_url"],
                "https://example.test/after",
            )
            self.assertEqual(len(agent._hist.calls), 1)
            image_url = agent._hist.calls[0][1]["image_url"]
            self.assertEqual(
                base64.b64decode(image_url.split(",", 1)[1]),
                b"POST",
            )

    async def test_v2_s_step_does_not_capture_or_attach_browser_screenshot(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            post_path = Path(tmp_dir) / "post.png"
            post_path.write_bytes(b"POST")

            agent = HybridBrowserAgent.__new__(HybridBrowserAgent)
            agent.web = _WebStub(post_path)
            agent._hist = _HistorySpy()
            record = {
                "step": 3,
                "active_mode_before": "S",
                "tool_call": {
                    "name": "search",
                    "arguments": {"query": ["example"]},
                },
                "observation": "search results",
                "report_parsed": {"think": "inspect the indexed sources"},
            }

            await agent._v2_add_executed_step_to_history(record)

            self.assertEqual(agent.web.calls, [])
            self.assertNotIn("post_shot_path", record)
            self.assertEqual(agent._hist.calls[0][1]["image_url"], "")

    async def test_v2_teacher_o_screenshot_is_stretched_without_changing_source(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pre_path = Path(tmp_dir) / "pre.png"
            post_path = Path(tmp_dir) / "post.png"
            Image.new("RGB", (1280, 720), color=(10, 20, 30)).save(pre_path)
            Image.new("RGB", (1280, 720), color=(40, 50, 60)).save(post_path)

            agent = HybridBrowserAgent.__new__(HybridBrowserAgent)
            agent.web = _WebStub(post_path)
            agent._hist = _HistorySpy()
            record = {
                "step": 8,
                "active_mode_before": "O",
                "img_path": str(pre_path),
                "tool_call": {
                    "name": "click",
                    "arguments": {"x": 500, "y": 500},
                },
                "observation": "clicked",
                "report_parsed": {"think": "click the centered control"},
            }

            with patch.dict(
                os.environ,
                {
                    "HYBRID_TEACHER_SCREENSHOT_SIZE": "1000",
                    "HYBRID_IMAGE_TRANSPORT": "base64",
                },
            ):
                await agent._v2_add_executed_step_to_history(record)

            image_url = agent._hist.calls[0][1]["image_url"]
            rendered_bytes = base64.b64decode(image_url.split(",", 1)[1])
            with Image.open(io.BytesIO(rendered_bytes)) as rendered:
                self.assertEqual(rendered.size, (1000, 1000))
            with Image.open(post_path) as original:
                self.assertEqual(original.size, (1280, 720))

    async def test_full_o_segment_keeps_every_post_action_screenshot(self):
        history = ReactHistory(max_o_screenshots=-1)
        history.render_multiturn_capped(
            "SYS",
            "O",
            "TASK",
            current_shot="data:image/png;base64,ENTRY",
        )
        for step in range(1, 5):
            history.add_step(
                "O",
                '{"name":"click","arguments":{"x":1,"y":1}}',
                f"observation-{step}",
                image_url=f"data:image/png;base64,POST{step}",
            )

        messages = history.render_multiturn_capped("SYS", "O", "TASK")
        user_image_counts = [
            sum(
                content.get("type") == "image_url"
                for content in message["content"]
            )
            for message in messages
            if message["role"] == "user"
        ]
        self.assertEqual(user_image_counts, [1, 1, 1, 1, 1])

    async def test_s_current_segment_is_text_only_but_prior_history_can_have_images(self):
        history = ReactHistory(max_o_screenshots=-1)
        history.add_step(
            "O",
            '{"name":"click","arguments":{"x":1,"y":1}}',
            "clicked",
            image_url="data:image/png;base64,O_POST",
        )
        # The history renderer groups duplicate action text, but segmentation
        # itself must not silently group away either screenshot.
        history.add_step(
            "O",
            '{"name":"click","arguments":{"x":1,"y":1}}',
            "clicked again",
            image_url="data:image/png;base64,O_POST_2",
        )
        history.add_step(
            "O",
            '{"name":"switch","arguments":{"reason":"need search"}}',
            "switched",
        )
        # Even if an upstream caller accidentally supplies an image for an S
        # step, the current S tool response must remain text-only.
        history.add_step(
            "S",
            '{"name":"search","arguments":{"query":["example"]}}',
            "results",
            image_url="data:image/png;base64,SHOULD_NOT_RENDER",
        )

        messages = history.render_multiturn_capped("SYS", "S", "TASK")
        head_images = [
            content["image_url"]["url"]
            for content in messages[1]["content"]
            if content.get("type") == "image_url"
        ]
        self.assertEqual(
            head_images,
            [
                "data:image/png;base64,O_POST",
                "data:image/png;base64,O_POST_2",
            ],
        )
        current_s_response = messages[-1]
        self.assertEqual(current_s_response["role"], "user")
        self.assertFalse(
            any(
                content.get("type") == "image_url"
                for content in current_s_response["content"]
            )
        )

    async def test_prior_screenshot_is_removed_only_after_its_step_is_folded(self):
        history = ReactHistory(max_o_screenshots=-1)
        history.add_step(
            "O",
            '{"name":"click","arguments":{"x":1,"y":1}}',
            "clicked",
            image_url="data:image/png;base64,O_POST",
        )
        history.add_step(
            "O",
            '{"name":"switch","arguments":{"reason":"need search"}}',
            "switched",
        )
        history.add_step(
            "S",
            '{"name":"search","arguments":{"query":["example"]}}',
            "results",
        )

        first_step = next(
            turn for turn in history.turns if turn.get("role") == "step"
        )
        history.folded.add(id(first_step))
        messages = history.render_multiturn_capped("SYS", "S", "TASK")

        self.assertFalse(
            any(
                content.get("type") == "image_url"
                for content in messages[1]["content"]
            )
        )
        # Folding is render-only; the source history remains recoverable.
        self.assertEqual(
            first_step["images"],
            ["data:image/png;base64,O_POST"],
        )


if __name__ == "__main__":
    unittest.main()
