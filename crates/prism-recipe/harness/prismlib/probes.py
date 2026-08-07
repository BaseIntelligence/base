"""G6 intermediate probes (recipe 1.3.0).

Every `PRISM_PROBE_EVERY`-th telemetry report (default 25) and while the
cumulative probe time budget (`PRISM_PROBE_TIME_BUDGET_S`, default 600 s)
lasts, the harness evaluates a fixed set of 8 short probe texts (from the
val family — the head of the frozen val cut) on the in-memory model under
`torch.no_grad()` + `model.eval()`, then restores `model.train()`. Each
point `{step, tokens_seen, wall_s, probe_loss}` is appended to the
telemetry state's `probe_curve`. A probe exception never kills training.
"""

import time

from .tokenizer import encode_tensor

PROBE_TEXT_COUNT = 8


def select_probe_texts(texts, train_rows):
    """Fixed short probe texts from the val family (deterministic)."""
    return list(texts[train_rows : train_rows + PROBE_TEXT_COUNT])


def teacher_forced_ce(model, tok, texts, device, seq_len):
    """Mean teacher-forced CE over the probe texts (probe-length truncated)."""
    import torch

    was_training = model.training
    model.eval()
    losses = []
    try:
        with torch.no_grad():
            for txt in texts:
                ids = encode_tensor(
                    tok, txt, device, truncation=True, max_length=seq_len
                )
                if ids.shape[1] < 2:
                    continue
                out = model(ids[:, :-1])
                logits = out.logits if hasattr(out, "logits") else out
                if logits.shape[1] < 1:
                    continue
                tgt = ids[:, 1:][:, -logits.shape[1] :]
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt.reshape(-1),
                    reduction="mean",
                )
                losses.append(loss.item())
    finally:
        if was_training:
            model.train()
    if not losses:
        raise RuntimeError("probe: no scored tokens")
    return sum(losses) / len(losses)


class ProbeRunner:
    """Every-K probe trigger with a cumulative wall-clock budget."""

    def __init__(
        self,
        model,
        stream,
        tok,
        texts,
        device,
        seq_len,
        every,
        time_budget_s,
        log,
    ):
        self.model = model
        self.stream = stream
        self.tok = tok
        self.texts = list(texts)
        self.device = device
        self.seq_len = int(seq_len)
        self.every = int(every)
        self.time_budget_s = float(time_budget_s)
        self.spent_s = 0.0
        self.log = log

    def maybe_probe(self, state, step):
        if self.every <= 0 or state["reports"] % self.every != 0:
            return
        if self.spent_s >= self.time_budget_s:
            return
        t0 = time.time()
        try:
            loss = teacher_forced_ce(
                self.model, self.tok, self.texts, self.device, self.seq_len
            )
        except Exception as exc:  # noqa: BLE001
            self.spent_s += time.time() - t0
            self.log(f"probe skipped at step {step}: {exc}")
            return
        self.spent_s += time.time() - t0
        state["probe_curve"].append(
            {
                "step": int(step),
                "tokens_seen": int(self.stream.tokens_seen),
                "wall_s": round(time.time() - state["t0"], 3),
                "probe_loss": float(loss),
            }
        )
