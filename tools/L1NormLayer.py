import torch


class L1NormLayer(torch.nn.Module):

    epsilon: float

    def __init__(self, epsilon: float = 10e-20) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input / (input.sum(dim=1, keepdim=True) + self.epsilon)
