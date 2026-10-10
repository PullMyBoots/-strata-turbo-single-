"""Local API: native llama.cpp mtmd video input, Strata generation, existing model alias and /tokenize."""
from __future__ import annotations

import copy
import json

from serve import server
from serve.frontend import IMAGE_PARTS
from serve import native_video

MAX_IMAGES = 64  # Strata's image cache keeps 64 embeddings alive while preparing a request.


def video_source(part):
    value = part.get(part["type"])
    if isinstance(value, dict):
        value = value.get("data") or value.get("url")
    if not isinstance(value, str) or not value:
        raise ValueError("input_video/video_url needs a data URL, URL or local path")
    return value


def normalize_request(req):
    messages = req.get("messages", [])
    parts = [p for msg in messages if isinstance(msg, dict) and isinstance(msg.get("content"), list)
             for p in msg["content"] if isinstance(p, dict)]
    if any(p.get("type") in ("input_audio", "audio") for p in parts):
        raise ValueError("audio input is not enabled by this backend")
    videos = sum(p.get("type") in ("input_video", "video_url") for p in parts)
    images = sum(p.get("type") in IMAGE_PARTS for p in parts)
    if not videos and images > MAX_IMAGES:
        raise ValueError("at most 64 images are supported per request")
    if not videos:
        return req
    out = copy.deepcopy(req)
    for msg in out.get("messages", []):
        if not isinstance(msg.get("content"), list):
            continue
        content = []
        for part in msg["content"]:
            if isinstance(part, dict) and part.get("type") in ("input_video", "video_url"):
                content.append({"type": "image_url", "image_url": {
                    "url": native_video.VIDEO_PREFIX + video_source(part)}})
            else:
                content.append(part)
        msg["content"] = content
    return out


_upstream_handler = server.make_handler


def make_handler(svc):
    class LocalHandler(_upstream_handler(svc)):
        def _openai(self, req):
            return super()._openai(normalize_request(req))

        def _json(self, code, obj):
            path = self.path.split("?")[0].rstrip("/")
            if code == 200 and path in ("/health", "/api/health"):
                obj = {**obj, "video_input": "native_llama_cpp_mtmd", "audio_input": False}
            elif code == 200 and path in ("/v1/models", "/models"):
                for item in obj.get("data", []):
                    modalities = item.setdefault("architecture", {}).setdefault("input_modalities", [])
                    if "video" not in modalities:
                        modalities.append("video")
            elif code == 200 and path == "/props":
                obj.setdefault("modalities", {}).update(video=True, audio=False)
            return super()._json(code, obj)

        def do_POST(self):
            if self.path.split("?")[0].rstrip("/") != "/tokenize":
                return super().do_POST()
            if not self._authorized() or self._foreign_page():
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if not isinstance(req, dict) or not isinstance(req.get("content"), str):
                    raise ValueError("content must be a string")
                if req.get("add_special"):
                    raise ValueError("this model has no automatic BOS token; use add_special=false")
                tokens = svc.tok.encode(req["content"], parse_special=req.get("parse_special", False))
                self._json(200, {"tokens": tokens})
            except (ValueError, TypeError) as exc:
                self._json(400, {"error": {"type": "invalid_request_error", "message": str(exc)}})
    if hasattr(svc, "request_slot"):
        from serve.concurrent import handler_with_slots
        return handler_with_slots(LocalHandler, svc)
    return LocalHandler


if __name__ == "__main__":
    native_video.install()
    server.make_handler = make_handler
    raise SystemExit(server.main())
