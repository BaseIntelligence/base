"""Minimal training.py fixture with sealed symbols (brief §6.4)."""


def num_floating_point_operations(args, batch_size):
    """FLOP accounting — sealed."""
    # keep formula stable
    return batch_size * args.seq_length * 2


def update_num_microbatches(consumed_samples, consistency_check=True):
    """GBS/MBS coherence — sealed."""
    return max(1, consumed_samples // 8)


def train_loop(args, iteration=0):
    consumed_train_samples = 0
    while iteration < args.train_iters:
        # legitimate optimizable body may surround seals
        iteration += 1
        consumed_train_samples += args.global_batch_size
    return consumed_train_samples
