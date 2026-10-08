"""Pack higher precision auxiliary MTP experts without modifying target model weights.

Read the already fetched BF16 tensors, use the pinned gguf-py quantizers, and
write [gate | up | down] GGUF blocks per expert for native_expert_grouped.
Dense weights and the draft vocabulary remain links to an existing runtime.
The target model still verifies all proposals. Acceptance and actual output
speed, rather than reconstruction error alone, decide whether to use this.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from _paths import add_gguf_py
add_gguf_py()
import gguf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="data/mtp directory with tensors and manifest")
    ap.add_argument("--base", type=Path, required=True, help="existing auxiliary runtime")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--format", choices=["q5_0", "q8_0"], required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {t["name"]: t for t in json.loads((args.src / "mtp-manifest.json").read_text())}
    gu = manifest["mtp.layers.0.mlp.experts.gate_up_proj"]
    down = manifest["mtp.layers.0.mlp.experts.down_proj"]
    ne, twice_ff, h = gu["shape"]
    ff = twice_ff // 2
    if down["shape"] != [ne, h, ff] or twice_ff != 2 * ff or h % 32 or ff % 32:
        raise ValueError("invalid expert geometry")
    provenance = []
    for item in [gu, down]:
        path = args.src / "tensors" / item["file"]
        if not path.exists():
            path = args.src / item["file"]
        digest = sha256(path)
        if digest != item["sha256"] or path.stat().st_size != int(np.prod(item["shape"])) * 2:
            raise ValueError(f"source checksum or size mismatch: {path}")
        provenance.append({"path": str(path.resolve()), "sha256": digest, "shape": item["shape"]})
    gu_raw = np.memmap(provenance[0]["path"], dtype="<u2", mode="r", shape=tuple(gu["shape"]))
    down_raw = np.memmap(provenance[1]["path"], dtype="<u2", mode="r", shape=tuple(down["shape"]))
    qtype = getattr(gguf.GGMLQuantizationType, args.format.upper())
    t0 = time.monotonic()
    errors = {"gate_up": [], "down": []}
    output_digest = hashlib.sha256()
    with (args.out / "experts.bin").open("xb") as stream:
        for expert in range(ne):
            for name, raw in [("gate_up", gu_raw), ("down", down_raw)]:
                weight = (np.asarray(raw[expert]).astype(np.uint32) << 16).view(np.float32)
                encoded = gguf.quants.quantize(weight, qtype)
                data = encoded.tobytes()
                stream.write(data)
                output_digest.update(data)
                if expert < 8:
                    decoded = gguf.quants.dequantize(encoded, qtype).reshape(weight.shape)
                    errors[name].append(float(np.linalg.norm(decoded - weight) / np.linalg.norm(weight)))
            if expert % 64 == 0:
                print(f"{args.format}: {expert + 1}/{ne} experts in {time.monotonic() - t0:.1f}s", flush=True)
    for name in ["dense.bin", "dense.txt", "draft_vocab.bin"]:
        (args.out / name).symlink_to((args.base / name).resolve(strict=True))
    # Write metadata last: a partial run must never look like a usable native runtime.
    (args.out / "experts-native.txt").write_text(f"strata-mtp-native 1 {qtype.value} {qtype.value} {h} {ff} {ne}\n")
    report = {"format": args.format, "ggml_type": qtype.value, "source": provenance,
              "experts_bytes": (args.out / "experts.bin").stat().st_size,
              "experts_sha256": output_digest.hexdigest(),
              "relative_rms_error_first_8": {k: float(np.mean(v)) for k, v in errors.items()},
              "seconds": time.monotonic() - t0, "target_weights_changed": False}
    (args.out / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
