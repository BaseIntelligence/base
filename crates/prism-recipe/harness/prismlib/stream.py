"""SeededTrainStream — harness-owned deterministic token-batch stream.

Iterates the pinned shard **minus the frozen val cut** (texts[:train_rows] +
texts[train_rows+val_rows:]), so the bpb val stays out of the train stream.
Order: seeded permutation per epoch (`random.Random(seed + epoch)`), docs
separated by EOS, windows of `seq_len + 1` tokens -> `(input_ids, labels)`
tensors on `ctx["device"]` of shape `(batch_size, seq_len)`; `labels` is
`input_ids` shifted by one position. The harness-side `tokens_seen` counter
counts yielded label tokens per batch — it is the authoritative token count
for METRICS_JSON v2 (`tokens_seen_source: "train_stream"`).

The stream is infinite: at epoch end it reshuffles with `seed + epoch` and
continues. Miners consume it with `for input_ids, labels in
ctx["train_stream"]:` plus their own stop condition (steps / guard).
"""

import random


class SeededTrainStream:
    def __init__(self, texts, tok, device, seq_len=512, batch_size=8, seed=0x00505249534D):
        self._texts = list(texts)
        if not self._texts:
            raise ValueError("empty train text pool")
        self._tok = tok
        self.device = device
        self.seq_len = max(8, int(seq_len))
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.tokens_seen = 0
        self.batches_yielded = 0
        self._epoch = 0
        self._order = self._perm(0)
        self._pos = 0
        self._buf = []
        self._eos = getattr(tok, "eos_token_id", None)

    def _perm(self, epoch):
        order = list(range(len(self._texts)))
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def _fill(self):
        need = self.batch_size * (self.seq_len + 1)
        while len(self._buf) < need:
            if self._pos >= len(self._order):
                self._epoch += 1
                self._order = self._perm(self._epoch)
                self._pos = 0
            text = self._texts[self._order[self._pos]]
            self._pos += 1
            ids = self._tok(text, add_special_tokens=False)["input_ids"]
            if not ids:
                continue
            self._buf.extend(ids)
            if self._eos is not None:
                self._buf.append(self._eos)

    def next_batch(self):
        import torch

        self._fill()
        need = self.batch_size * (self.seq_len + 1)
        window = self._buf[:need]
        del self._buf[:need]
        ids = torch.tensor(window, dtype=torch.long).view(self.batch_size, self.seq_len + 1)
        input_ids = ids[:, :-1].contiguous().to(self.device)
        labels = ids[:, 1:].contiguous().to(self.device)
        self.tokens_seen += labels.numel()
        self.batches_yielded += 1
        return input_ids, labels

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()
