"""Toy model — baseline architecture surface for fixture patches."""


class ToyModel:
    def __init__(self, width: int = 8) -> None:
        self.width = width

    def forward(self, x):
        return x
