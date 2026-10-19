import unittest

from react_v2_prompts import system_prompt


class ReactV2PromptTest(unittest.TestCase):
    def test_human_verification_is_terminal_on_operation_side(self):
        prompt = system_prompt("O")

        self.assertIn("verify you are human", prompt)
        self.assertIn("Human verification is a terminal condition", prompt)
        self.assertIn("End the entire task with `<answer>`", prompt)
        self.assertIn("Do not click, wait, refresh, retry", prompt)
        self.assertIn("`switch` to another path", prompt)

    def test_human_verification_rule_is_not_added_to_search_side(self):
        self.assertNotIn(
            "Human verification is a terminal condition",
            system_prompt("S"),
        )


if __name__ == "__main__":
    unittest.main()
