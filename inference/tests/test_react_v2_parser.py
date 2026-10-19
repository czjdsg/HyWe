import unittest

from react_v2 import parse_output


class ReactV2ParserTest(unittest.TestCase):
    def test_accepts_gateway_tool_name_followed_by_arguments(self):
        raw = (
            '<tool_call>search\n'
            '{"query": ["Thanksgiving 2026 date", "official holiday date"]}'
        )

        parsed = parse_output(raw)

        self.assertTrue(parsed["ok"])
        self.assertTrue(parsed["repaired"])
        self.assertEqual(
            parsed["tool_call"],
            {
                "name": "search",
                "arguments": {
                    "query": [
                        "Thanksgiving 2026 date",
                        "official holiday date",
                    ]
                },
            },
        )

    def test_accepts_leading_reasoning_and_closed_gateway_tool_call(self):
        raw = (
            "I need to establish the date first."
            '<tool_call>visit\n'
            '{"url": "https://example.com", "goal": "find the date"}'
            "</tool_call>"
        )

        parsed = parse_output(raw)

        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["tool_call"]["name"], "visit")
        self.assertEqual(
            parsed["tool_call"]["arguments"]["goal"],
            "find the date",
        )

    def test_accepts_partially_stripped_arg_value_format(self):
        raw = (
            '<tool_call>search\n'
            'query: ["TomTom route distance toll avoid"]'
            "</arg_value></tool_call>"
        )

        parsed = parse_output(raw)

        self.assertTrue(parsed["ok"])
        self.assertEqual(
            parsed["tool_call"],
            {
                "name": "search",
                "arguments": {
                    "query": ["TomTom route distance toll avoid"],
                },
            },
        )

    def test_accepts_native_arg_key_arg_value_pairs(self):
        raw = (
            '<tool_call>visit\n'
            "<arg_key>url</arg_key>"
            "<arg_value>\"https://example.com\"</arg_value>"
            "<arg_key>goal</arg_key>"
            "<arg_value>\"find the live price\"</arg_value>"
            "</tool_call>"
        )

        parsed = parse_output(raw)

        self.assertTrue(parsed["ok"])
        self.assertEqual(
            parsed["tool_call"]["arguments"],
            {
                "url": "https://example.com",
                "goal": "find the live price",
            },
        )

    def test_accepts_kimi_native_tool_call_markers(self):
        raw = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.search:2"
            "<|tool_call_argument_begin|>"
            '{"query": ["TomTom route planner", "avoid toll roads"]}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )

        parsed = parse_output(raw)

        self.assertTrue(parsed["ok"])
        self.assertTrue(parsed["repaired"])
        self.assertEqual(
            parsed["tool_call"],
            {
                "name": "search",
                "arguments": {
                    "query": ["TomTom route planner", "avoid toll roads"],
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
