"""Video API setting passthrough regression tests; no GPU required."""
import copy
import unittest

from serve.local_api import normalize_request
from serve.server import request_timings


class VideoSettingsPassthrough(unittest.TestCase):
    def request(self, **extra):
        return {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Review the video."},
            {"type": "video_url", "video_url": {"url": "/tmp/test.mp4"}}]}], **extra}

    def test_review_preserves_requested_settings_without_adding_budget(self):
        request = self.request(max_tokens=2000, reasoning_effort="xhigh")
        original = copy.deepcopy(request)
        normalized = normalize_request(request)
        self.assertNotIn("reasoning_budget_tokens", normalized)
        self.assertEqual(normalized["max_tokens"], 2000)
        self.assertEqual(normalized["reasoning_effort"], "xhigh")
        self.assertEqual(request, original)

    def test_stream_completion_limits_are_preserved(self):
        request = self.request(max_completion_tokens=512, max_tokens=2000, stream=True)
        normalized = normalize_request(request)
        self.assertNotIn("reasoning_budget_tokens", normalized)
        self.assertEqual(normalized["max_completion_tokens"], 512)
        self.assertEqual(normalized["max_tokens"], 2000)

    def test_tiny_output_keeps_the_requested_thinking_level(self):
        request = self.request(max_tokens=32, chat_template_kwargs={"reasoning_effort": "xhigh"})
        self.assertEqual(normalize_request(request)["chat_template_kwargs"], {"reasoning_effort": "xhigh"})

    def test_explicit_budget_including_disabled_is_preserved(self):
        for budget in (0, 64, 1200, "invalid"):
            request = self.request(max_tokens=2000, reasoning_budget_tokens=budget)
            self.assertEqual(normalize_request(request)["reasoning_budget_tokens"], budget)

    def test_unlimited_and_large_completions_have_no_automatic_budget(self):
        for limit in (None, 0, -1, 32768):
            request = self.request() if limit is None else self.request(max_tokens=limit)
            self.assertNotIn("reasoning_budget_tokens", normalize_request(request))

    def test_text_request_is_unchanged(self):
        request = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 2000}
        self.assertIs(normalize_request(request), request)

    def test_continuation_cached_count_stays_within_input_tokens(self):
        stats = request_timings(100, 20, {"reused": 120, "generated": 10,
                                         "prompt_ms": 1.0, "decode_ms": 10.0})
        self.assertEqual((stats["cache_n"], stats["prompt_n"]), (100, 0))


if __name__ == "__main__":
    unittest.main()
