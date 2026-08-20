# loopmoe (retired)

This directory is **not** the Prism recipe 2.1 reference.

Fine-grained MoE / LoopMoE at ~1B wastes MFU on 4×5090 (tiny expert GEMMs,
irregular routing, no NVFP4 wgrad). Copy
[`../dense-1b/`](../dense-1b/) instead: a **dense ~975M** transformer
(GQA + SwiGLU + ZeRO-1).

MoE remains a valid miner experiment if you write your own pack. Do not
submit this folder. Do not point anything here at live `:28092`.
