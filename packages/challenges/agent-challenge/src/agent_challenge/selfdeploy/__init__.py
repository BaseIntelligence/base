"""Miner-facing self-deploy flow for the canonical Phala eval image (architecture §4 C7).

The miner self-deploys and funds a Phala **Intel TDX CPU** CVM running the
canonical, digest-pinned eval image; the validator/subnet keeps the trust root
(measurement allowlist + golden key-release + quote verification). This package
implements the miner-facing command surface around the already-landed building
blocks (``canonical.compose``/``canonical.measurement``/``canonical.report_data``/
``keyrelease``): fetch/prepare the image + generated compose, publish/reproduce
the canonical measurement, deploy a CPU-only CVM (with money/GPU guards + a
no-spend dry-run), point the run at the validator key-release endpoint, and
surface the attested result for allowlist checking.

Money/GPU safety is enforced BEFORE any provisioning: GPU targets are refused,
over-cap shapes are refused, and missing credentials error clearly without a
Phala mutate call (AGENTS.md Mission Boundaries).
"""
