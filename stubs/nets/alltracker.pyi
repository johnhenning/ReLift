from typing import Any

import torch

class Net(torch.nn.Module):
    def __init__(self, window: int, init_weights: bool = True) -> None: ...
    def forward(
        self, x: torch.Tensor, iters: int = 4, sw: Any = None, is_training: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, Any, Any]: ...
