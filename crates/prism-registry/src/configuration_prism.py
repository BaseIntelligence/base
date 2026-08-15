"""Prism custom-arch Hub config (trust_remote_code)."""

from transformers import PretrainedConfig


class PrismConfig(PretrainedConfig):
    model_type = "prism_custom"

    def __init__(self, prism_arch_id=None, checkpoint_file="checkpoint.pt", **kwargs):
        super().__init__(**kwargs)
        self.prism_arch_id = prism_arch_id
        self.checkpoint_file = checkpoint_file
