"""G6 intermediate probes (recipe 1.3.0; byte/compute curve in v2.1).

The harness probes before and after train and every `PRISM_PROBE_EVERY`-th
*accounted stream batch* (default 25). The cadence is organizer-owned, not a
miner-controlled telemetry-report counter. A fixed set of 8 short probe texts
from the frozen val family is evaluated under `torch.no_grad()` + eval mode,
then the prior train/eval mode is restored. Each point carries token, UTF-8
byte, and attested-FLOPs coordinates. A probe exception never kills training.
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
        self.probe_bytes_per_token = self._probe_compression()

    def _probe_compression(self):
        """UTF-8 bytes per predicted token on the fixed probe texts."""
        n_bytes = 0.0
        n_tokens = 0
        for text in self.texts:
            try:
                ids = self.tok(text, add_special_tokens=False)["input_ids"]
            except Exception:  # noqa: BLE001
                continue
            predicted = min(len(ids), self.seq_len) - 1
            if predicted <= 0:
                continue
            # Probe texts are selected short; retain a proportional fallback
            # if a submitted tokenizer expands one beyond the sequence cap.
            kept = min(1.0, (predicted + 1) / max(1, len(ids)))
            n_bytes += len(text.encode("utf-8", "ignore")) * kept
            n_tokens += predicted
        return (n_bytes / n_tokens) if n_tokens > 0 else 0.0

    def maybe_probe(self, state, step, force=False):
        batch_no = int(getattr(self.stream, "batches_yielded", 0))
        if not force and (self.every <= 0 or batch_no <= 0 or batch_no % self.every != 0):
            return
        if not force and self.spent_s >= self.time_budget_s:
            return
        tokens_seen = int(getattr(self.stream, "tokens_seen", 0))
        flops_spent = float(getattr(self.stream, "flops_per_token", 0.0)) * tokens_seen
        for prior in state["probe_curve"]:
            if (
                int(prior.get("tokens_seen", -1)) == tokens_seen
                and float(prior.get("flops_spent", -1.0)) == flops_spent
            ):
                return
        if not self.texts:
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
        # Byte + FLOPs coordinates alongside the token coordinate. The y-axis
        # conversion uses the fixed PROBE texts' compression ratio (not train
        # corpus compression); the x-axis is harness-accounted train bytes.
        bpt = self.probe_bytes_per_token
        accounted_bytes = (
            self.stream.accounted_bytes_seen()
            if hasattr(self.stream, "accounted_bytes_seen")
            else int(getattr(self.stream, "bytes_seen", 0))
        )
        point = {
            "step": int(step),
            "tokens_seen": tokens_seen,
            # log10 needs a positive origin; one byte represents the genuine
            # pre-train measurement at zero consumed train bytes.
            "bytes_seen": max(1, int(accounted_bytes)),
            "bytes_per_token": float(bpt),
            "flops_spent": flops_spent,
            "wall_s": round(time.time() - state["t0"], 3),
            "probe_loss": float(loss),
        }
        # bits/byte = (nats/token) / ln2 * (tokens/byte); emitted here so the
        # G6 curve module can score a tokenizer-neutral quantity directly.
        if bpt > 0:
            point["probe_bits_per_byte"] = float(loss) / (0.6931471805599453 * bpt)
        state["probe_curve"].append(point)

    def force_probe(self, state, step):
        """Required boundary measurement, independent of cadence/budget."""
        self.maybe_probe(state, step, force=True)
