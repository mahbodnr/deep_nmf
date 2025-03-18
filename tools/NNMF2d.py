import torch


class NNMF2d(torch.nn.Module):

    in_channels: int
    out_channels: int
    weight: torch.Tensor
    iterations: int
    epsilon: float | None
    init_min: float
    init_max: float
    local_learning: bool
    local_learning_kl: bool

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        device=None,
        dtype=None,
        iterations: int = 20,
        epsilon: float | None = None,
        init_min: float = 0.0,
        init_max: float = 1.0,
        local_learning: bool = False,
        local_learning_kl: bool = False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.init_min = init_min
        self.init_max = init_max

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.iterations = iterations
        self.local_learning = local_learning
        self.local_learning_kl = local_learning_kl

        self.weight = torch.nn.parameter.Parameter(
            torch.empty((out_channels, in_channels), **factory_kwargs)
        )

        self.reset_parameters()
        self.functional_nnmf2d = FunctionalNNMF2d.apply

        self.epsilon = epsilon

    def extra_repr(self) -> str:
        s: str = f"{self.in_channels}, {self.out_channels}"

        if self.epsilon is not None:
            s += f", epsilon={self.epsilon}"
        s += f", local_learning={self.local_learning}"

        if self.local_learning:
            s += f", local_learning_kl={self.local_learning_kl}"

        return s

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight, a=self.init_min, b=self.init_max)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        positive_weights = torch.abs(self.weight)
        positive_weights = positive_weights / (
            positive_weights.sum(dim=1, keepdim=True) + 10e-20
        )

        h_dyn = self.functional_nnmf2d(
            input,
            positive_weights,
            self.out_channels,
            self.iterations,
            self.epsilon,
            self.local_learning,
            self.local_learning_kl,
        )

        return h_dyn


class FunctionalNNMF2d(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        out_channels: int,
        iterations: int,
        epsilon: float | None,
        local_learning: bool,
        local_learning_kl: bool,
    ) -> torch.Tensor:

        # Prepare h
        h = torch.full(
            (input.shape[0], out_channels, input.shape[-2], input.shape[-1]),
            1.0 / float(out_channels),
            device=input.device,
            dtype=input.dtype,
        )

        h = h.movedim(1, -1)
        input = input.movedim(1, -1)
        for _ in range(0, iterations):
            reconstruction = torch.nn.functional.linear(h, weight.T)
            reconstruction += 1e-20
            if epsilon is None:
                h *= torch.nn.functional.linear((input / reconstruction), weight)
            else:
                h *= 1 + epsilon * torch.nn.functional.linear(
                    (input / reconstruction), weight
                )
            h /= h.sum(-1, keepdim=True) + 10e-20
        h = h.movedim(-1, 1)
        input = input.movedim(-1, 1)

        # ###########################################################
        # Save the necessary data for the backward pass
        # ###########################################################
        ctx.save_for_backward(input, weight, h)
        ctx.local_learning = local_learning
        ctx.local_learning_kl = local_learning_kl

        assert torch.isfinite(h).all()
        return h

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor) -> tuple[  # type: ignore
        torch.Tensor,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:

        # ##############################################
        # Default values
        # ##############################################
        grad_weight: torch.Tensor | None = None

        # ##############################################
        # Get the variables back
        # ##############################################
        (input, weight, h) = ctx.saved_tensors

        # The back prop gradient
        h = h.movedim(1, -1)
        grad_output = grad_output.movedim(1, -1)
        input = input.movedim(1, -1)
        big_r = torch.nn.functional.linear(h, weight.T)
        big_r_div = 1.0 / (big_r + 1e-20)

        factor_x_div_r = input * big_r_div

        grad_input: torch.Tensor = (
            torch.nn.functional.linear(h * grad_output, weight.T) * big_r_div
        )

        del big_r_div

        # The weight gradient
        if ctx.local_learning is False:
            del big_r

            grad_weight = -torch.nn.functional.linear(
                h.reshape(
                    grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                    h.shape[3],
                ).T,
                (factor_x_div_r * grad_input)
                .reshape(
                    grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                    grad_input.shape[3],
                )
                .T,
            )

            grad_weight += torch.nn.functional.linear(
                (h * grad_output)
                .reshape(
                    grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                    h.shape[3],
                )
                .T,
                factor_x_div_r.reshape(
                    grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                    grad_input.shape[3],
                ).T,
            )

        else:
            if ctx.local_learning_kl:
                grad_weight = -torch.nn.functional.linear(
                    h.reshape(
                        grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                        h.shape[3],
                    ).T,
                    factor_x_div_r.reshape(
                        grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                        grad_input.shape[3],
                    ).T,
                )
            else:
                grad_weight = -torch.nn.functional.linear(
                    h.reshape(
                        grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                        h.shape[3],
                    ).T,
                    (2 * (input - big_r))
                    .reshape(
                        grad_input.shape[0] * grad_input.shape[1] * grad_input.shape[2],
                        grad_input.shape[3],
                    )
                    .T,
                )
        grad_input = grad_input.movedim(-1, 1)
        assert torch.isfinite(grad_input).all()
        assert torch.isfinite(grad_weight).all()

        return (
            grad_input,
            grad_weight,
            None,
            None,
            None,
            None,
            None,
        )
