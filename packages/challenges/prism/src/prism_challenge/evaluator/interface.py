from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from .param_ladder import (
    EXPLORE_MAX_PARAMETERS,
    STAGE_EXPLORE,
    max_parameters_for_stage,
    normalize_param_ladder_stage,
)

# Two-script submission contract (architecture.md section 2). A v2 bundle MUST contain TWO
# distinct scripts: an architecture module exposing a `build_model(ctx)` factory and a training
# module exposing a `train(ctx)` entrypoint. The legacy single-module re-export idiom (both hooks
# in one combined module) no longer satisfies the contract.
DEFAULT_ARCHITECTURE_ENTRYPOINT = "architecture.py"
DEFAULT_TRAINING_ENTRYPOINT = "training.py"
ARCHITECTURE_FACTORY_NAME = "build_model"
TRAINING_ENTRYPOINT_NAME = "train"


class SubmissionContractError(ValueError):
    """Raised when a submission bundle violates the two-script submission contract."""


@dataclass(frozen=True)
class PrismContext:
    vocab_size: int = 4096
    sequence_length: int = 128
    max_layers: int = 96
    # Dual ladder default = explore (124M). Promote-stage runs set 350M + stage="promote".
    max_parameters: int = EXPLORE_MAX_PARAMETERS
    # Ladder stage label surfacing on scores/manifests (explore | promote).
    param_ladder_stage: Literal["explore", "promote"] = STAGE_EXPLORE
    seed: int = 1337
    checkpoint_dir: Path | None = None
    resume_checkpoint_dir: Path | None = None
    checkpoint_api_version: int = 1
    attempt: int = 1
    is_resume: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    distributed_backend: str | None = None
    device: str = "cpu"
    checkpoint_metadata: dict[str, object] = field(default_factory=dict)
    # v2 two-script contract fields (architecture.md section 2). The challenge owns the data and
    # forces the seed/init; the miner's train(ctx) loop reads only the read-only locked train
    # split at ``data_dir`` and writes only under ``artifacts_dir``.
    data_dir: str | None = None
    artifacts_dir: str | None = None
    token_budget: int | None = None
    step_budget: int | None = None
    reference_tokenizer_dir: str | None = None

    @property
    def max_seq_len(self) -> int:
        return self.sequence_length

    @property
    def max_params(self) -> int:
        return self.max_parameters

    @property
    def ladder_stage(self) -> Literal["explore", "promote"]:
        """Canonical dual-ladder stage for this evaluation context."""
        return normalize_param_ladder_stage(self.param_ladder_stage)

    @property
    def ladder_stage_cap(self) -> int:
        """Hard stage cap (ignores per-run numeric overrides on ``max_parameters``)."""
        return max_parameters_for_stage(self.ladder_stage)

    def reference_tokenizer(self, name: str) -> Any:
        """Load a pre-staged reference tokenizer (offline, no network).

        The reference tokenizers (gpt2 via a tiktoken cache, llama via a sentencepiece ``.model``)
        are baked into the eval image and/or staged on the read-only data volume. This resolves the
        staged path and loads it lazily; it never reaches the network. When the dir is unset the
        staging path is resolved from ``PRISM_REFERENCE_TOKENIZER_DIR`` (baked image).
        """
        from .reference_tokenizers import load_reference_tokenizer

        return load_reference_tokenizer(name, self.reference_tokenizer_dir)


@dataclass(frozen=True)
class TrainingRecipe:
    learning_rate: float = 3e-4
    batch_size: int = 4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    weight_decay: float = 0.01


@dataclass(frozen=True)
class PrismBatch:
    tokens: Any
    targets: Any | None = None
    metadata: dict[str, Any] | None = None


class PrismModelModule(Protocol):
    def build_model(self, ctx: PrismContext) -> Any: ...

    def get_recipe(self, ctx: PrismContext) -> TrainingRecipe: ...

    def configure_optimizer(self, model: Any, recipe: TrainingRecipe, ctx: PrismContext) -> Any: ...

    def inference_logits(self, model: Any, batch: PrismBatch, ctx: PrismContext) -> Any: ...

    def compute_loss(self, model: Any, batch: PrismBatch, ctx: PrismContext) -> Any: ...

    def train_step(
        self, model: Any, batch: PrismBatch, optimizer: Any, ctx: PrismContext
    ) -> Any: ...

    def save_checkpoint(
        self, model: Any, checkpoint_dir: Path, ctx: PrismContext
    ) -> str | dict[str, object] | None: ...

    def load_checkpoint(
        self, model: Any, checkpoint_dir: Path, ctx: PrismContext
    ) -> dict[str, object] | None: ...


def import_torch() -> Any:
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for Prism evaluation") from exc
