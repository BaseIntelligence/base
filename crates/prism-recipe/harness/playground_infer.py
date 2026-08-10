#!/usr/bin/env python3
"""Operator playground inference: load parked checkpoint + emit text/logprobs.

Invoked by `prism-playground` as:
  python3 playground_infer.py /path/to/request.json

request.json fields: prompt, max_tokens, temperature, return_logprobs,
top_logprobs, checkpoint, workdir (architecture.py + training.py), metadata.

Writes a single JSON object to stdout. Fail-closed on missing torch / bad ckpt.
"""

from __future__ import annotations

import json
import os
import sys
import time


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: playground_infer.py request.json", file=sys.stderr)
        return 2
    req = json.loads(open(argv[1], encoding="utf-8").read())
    workdir = req["workdir"]
    ckpt = req["checkpoint"]
    sys.path.insert(0, workdir)
    # Prefer the harness package next to this script when present.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    t0 = time.time()
    try:
        import torch
    except ImportError as exc:
        print(json.dumps({"error": f"torch unavailable: {exc}"}))
        return 1

    # Detect Sim stub early (also checked in Rust).
    with open(ckpt, "rb") as f:
        head = f.read(32)
    if head.startswith(b"PRISM_SIM_CKPT"):
        print(
            json.dumps(
                {
                    "model": {
                        "kind": req.get("kind"),
                        "submission_id": req.get("submission_id"),
                        "arch_id": req.get("arch_id"),
                        "bpb": req.get("bpb"),
                        "repo_path": "top-model",
                    },
                    "text": "",
                    "tokens": [],
                    "logprobs": [],
                    "diagnostics": {"sim_stub": True, "checkpoint": ckpt},
                }
            )
        )
        return 0

    import importlib.util
    from prismlib import tokenizer as tok_mod

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    device = "cuda" if torch.cuda.is_available() else "cpu"
    arch_path = os.path.join(workdir, "architecture.py")
    train_path = os.path.join(workdir, "training.py")
    arch = _load(arch_path, "prism_play_arch")
    train = _load(train_path, "prism_play_train") if os.path.isfile(train_path) else None
    ctx = {
        "device": device,
        "workdir": workdir,
        "max_seq_len": 2048,
        "vocab_size": None,
    }
    tok, tok_spec = tok_mod.resolve(ctx, device, arch_mod=arch, train_mod=train)
    ctx["tokenizer"] = tok
    ctx["vocab_size"] = tok_spec["vocab_size"]
    if not hasattr(arch, "build_model"):
        raise RuntimeError("build_model missing")
    model = arch.build_model(ctx)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return nn.Module")
    model.to(device)
    model.eval()
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    prompt = req["prompt"]
    max_tokens = int(req.get("max_tokens") or 64)
    temperature = float(req.get("temperature") or 0.0)
    top_k = int(req.get("top_logprobs") or 5)
    return_logprobs = bool(req.get("return_logprobs", True))

    ids = tok_mod.encode_tensor(tok, prompt, device=device)
    generated = []
    logprob_rows = []
    ttft_ms = None
    t_gen0 = time.time()
    with torch.no_grad():
        for step in range(max_tokens):
            out = model(ids)
            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            if temperature and temperature > 0:
                logits_s = logits / temperature
                next_id = torch.multinomial(torch.softmax(logits_s.float(), dim=-1), 1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            nid = int(next_id.item())
            generated.append(nid)
            if return_logprobs:
                lp = torch.log(probs[0, nid] + 1e-12).item()
                topv, topi = torch.topk(probs[0], k=min(top_k, probs.shape[-1]))
                top = []
                for v, i in zip(topv.tolist(), topi.tolist()):
                    tok_s = tok.decode([int(i)]) if hasattr(tok, "decode") else str(i)
                    top.append({"token": tok_s, "token_id": int(i), "logprob": float(__import__("math").log(v + 1e-12))})
                tok_s = tok.decode([nid]) if hasattr(tok, "decode") else str(nid)
                logprob_rows.append(
                    {"token": tok_s, "token_id": nid, "logprob": float(lp), "top": top}
                )
            if ttft_ms is None:
                ttft_ms = (time.time() - t_gen0) * 1000.0
            ids = torch.cat([ids, next_id], dim=1)
            eos = getattr(tok, "eos_token_id", None)
            if eos is not None and nid == int(eos):
                break

    text = tok.decode(generated) if hasattr(tok, "decode") else ""
    elapsed = time.time() - t0
    n_gen = len(generated)
    tpot = ((elapsed * 1000.0) - (ttft_ms or 0.0)) / max(1, n_gen - 1) if n_gen > 1 else None
    print(
        json.dumps(
            {
                "model": {
                    "kind": req.get("kind"),
                    "submission_id": req.get("submission_id"),
                    "arch_id": req.get("arch_id"),
                    "bpb": req.get("bpb"),
                    "repo_path": "top-model",
                },
                "text": text,
                "tokens": generated,
                "logprobs": logprob_rows,
                "diagnostics": {
                    "n_prompt_tokens": int(ids.shape[1]) - n_gen,
                    "n_generated": n_gen,
                    "ttft_ms": ttft_ms,
                    "tpot_ms": tpot,
                    "device": device,
                    "checkpoint": ckpt,
                    "wall_s": elapsed,
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
