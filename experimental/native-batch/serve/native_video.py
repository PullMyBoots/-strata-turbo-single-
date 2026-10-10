"""Bridge llama.cpp mtmd's native video prompt into Strata GENI/SVE2."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import uuid
from pathlib import Path

from serve import server
from serve.frontend import images_of

VIDEO_PREFIX = "strata-video:"
MEDIA_MARKER = "<__strata_native_media__>"
_upstream_prepare = server.Service.prepare


def native_prepare(svc, messages, tools, kwargs, max_new=None, force=None, req=None):
    sources = images_of(messages)
    if not any(src.startswith(VIDEO_PREFIX) for src in sources):
        return _upstream_prepare(svc, messages, tools, kwargs, max_new, force=force, req=req)
    svc.embeddings.path = None
    if svc.vision is None:
        raise ValueError("video requires the native vision encoder")
    vision = svc.vision
    rendered_messages = copy.deepcopy(messages)
    # Let mtmd insert each media's own vision wrappers and timestamp tokens, as llama-server does.
    for msg in rendered_messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image":
                    item.clear()
                    item.update(type="text", text=MEDIA_MARKER)
                elif MEDIA_MARKER in item.get("text", ""):
                    raise ValueError("text contains the reserved native media marker")
        elif MEDIA_MARKER in (msg.get("content") or ""):
            raise ValueError("text contains the reserved native media marker")
    prompt = svc.template.render(rendered_messages, tools=tools, **kwargs)
    if prompt.count(MEDIA_MARKER) != len(sources):
        raise ValueError("native video media order does not match the rendered prompt")
    with svc.fifo, vision.lock, tempfile.TemporaryDirectory(prefix="strata-native-media-") as folder:
        cfg = getattr(vision, "native_video_settings", {})
        cache = getattr(vision, "native_prompt_cache", None)
        if cache is None:
            cache = vision.native_prompt_cache = {}
        digest = hashlib.sha256(prompt.encode())
        digest.update(json.dumps(cfg, sort_keys=True).encode())
        paths = []
        for i, source in enumerate(sources):
            is_video = source.startswith(VIDEO_PREFIX)
            source = source[len(VIDEO_PREFIX):] if is_video else source
            try:
                data = vision.load(source)
                if not is_video:
                    data = vision.normalize(data)
            except (OSError, ValueError) as exc:
                raise ValueError("native media source could not be read") from exc
            digest.update(hashlib.sha256(data).digest())
            path = Path(folder) / f"media-{i}.{'video' if is_video else 'img'}"
            path.write_bytes(data)
            paths.append(str(path))
        key = digest.hexdigest()
        cached = cache.get(key)
        if cached is None:
            output = vision.dir / f"native-{key}.sve"
            request_file = Path(folder) / "request.json"
            request_file.write_text(json.dumps(dict(prompt=prompt, media=paths,
                fps=cfg.get("fps", 4.0), timestamp_interval_ms=cfg.get("timestamp_interval_ms", 5000),
                max_context=svc.engine.max_context-server.CTX_SLACK)))
            try:
                vision.proc.stdin.write(f"PRE {request_file} {output}\n")
                vision.proc.stdin.flush()
                line = vision.proc.stdout.readline().strip()
                if not line.startswith("OKPRE "):
                    raise ValueError("native video encoding failed: " + line.removeprefix("ERR "))
                meta = json.loads(Path(str(output)+".json").read_text())
            except BaseException:
                output.unlink(missing_ok=True)
                Path(str(output)+".json").unlink(missing_ok=True)
                raise
            cached = (output, meta)
            cache[key] = cached
            if len(cache) > 2:
                oldest = next(iter(cache))
                old_file, _ = cache.pop(oldest)
                old_file.unlink(missing_ok=True)
                Path(str(old_file)+".json").unlink(missing_ok=True)
        output, meta = cached
        ids = list(meta["ids"])
        if force and kwargs.get("enable_thinking", True) is False:
            ids += svc.tok.encode(force, parse_special=True)
        room = svc.engine.max_context - server.CTX_SLACK - len(ids)
        if room < 1:
            raise ValueError("native video prompt leaves no room to answer")
        if max_new is None or max_new <= 0:
            max_new = room
        elif max_new > room:
            if not svc.fit_max_tokens:
                raise ValueError(f"native video prompt ({len(ids)} tokens) plus output exceeds context; max_tokens at most {room}")
            max_new = room
        combined = vision.dir / f"req-{uuid.uuid4().hex}.sve"
        shutil.copyfile(output, combined)
        svc.embeddings.path = combined
        print(f"[strata] native mtmd video: {len(ids)} prompt tokens, {len(meta['segments'])} fused image chunks", flush=True)
        if req is not None and req.get("strata_prefix") is not None:
            req["strata_prefix"] = svc.resolve_prefix(req["strata_prefix"], messages, tools, kwargs, ids, True)
        return ids, kwargs.get("enable_thinking", True) is not False, max_new


_upstream_vision_init = server.Vision.__init__


def vision_init(self, cfg, *args, **kwargs):
    _upstream_vision_init(self, cfg, *args, **kwargs)
    self.native_video_settings = cfg.get("native_video", {})


def install():
    server.Service.prepare = native_prepare
    server.Vision.__init__ = vision_init
