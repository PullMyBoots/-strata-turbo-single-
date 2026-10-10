"""Claude Code cache stability and pinned-upstream prompt parity."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

from serve.frontend import anthropic_to_messages, pin_billing_stamp, ChatTemplate

ROOT = Path(__file__).resolve().parents[1]


class BillingStampTests(unittest.TestCase):
    def test_random_stamps_do_not_change_system_instructions(self):
        header = "x-anthropic-billing-header: cc_version=2.1.290.{}; cch={};"
        values = [header.format("e85", "abcde"), header.format("473", "12345")]
        prompts = []
        for value in values:
            req = {"system": [{"type": "text", "text": value},
                              {"type": "text", "text": "Keep every instruction."}],
                   "messages": [{"role": "user", "content": value}]}
            original = copy.deepcopy(req)
            messages, _, _ = anthropic_to_messages(req)
            prompts.append(messages[0]["content"])
            self.assertEqual(messages[1]["content"], value)
            self.assertEqual(req, original)
        self.assertEqual(prompts[0], prompts[1])
        self.assertTrue(prompts[0].endswith("Keep every instruction."))

    def test_plain_or_out_of_bounds_text_is_preserved(self):
        texts = ["You are helpful. cch=abc; cc_version=1.2.3.4;",
                 "Hello.\nx-anthropic-billing-header: cch=abc;",
                 "x-anthropic-billing-header: cc_version=2.1.290;",
                 "x-anthropic-billing-header: cch=" + "a" * 17 + ";",
                 "x-anthropic-billing-header: cc_version=2.1.290.e85",
                 "x-anthropic-billing-header: " + "x" * 161 + "cch=abc;"]
        for text in texts:
            self.assertEqual(pin_billing_stamp(text), text)

    def test_real_harness_prompt_matches_pinned_adapter(self):
        import json
        path = ROOT / "experimental/native-batch/serve/frontend.py"
        spec = importlib.util.spec_from_file_location("billing_upstream_reference", path)
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)
        req = json.loads((ROOT / "evidence/showcase/pelican/initial-request.json").read_text())
        template = ChatTemplate(ROOT / "serve/chat_template.jinja")
        actual = anthropic_to_messages(req)
        expected = reference.anthropic_to_messages(req)
        self.assertEqual(actual, expected)
        self.assertEqual(template.render(actual[0], tools=actual[1], **actual[2]),
                         template.render(expected[0], tools=expected[1], **expected[2]))


if __name__ == "__main__":
    unittest.main()
