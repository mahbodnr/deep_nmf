import torch
from tools.L1NormLayer import L1NormLayer
from tools.NNMF2d import NNMF2d
from tools.append_parameter import append_parameter


def append_block(
    network: torch.nn.Sequential,
    number_of_neurons_a: int,
    number_of_neurons_b: int,
    test_image: torch.Tensor,
    parameter_neuron_a: list[torch.nn.parameter.Parameter],
    parameter_neuron_b: list[torch.nn.parameter.Parameter],
    parameter_batchnorm2d: list[torch.nn.parameter.Parameter],
    device: torch.device,
    dilation: tuple[int, int] | int = 1,
    padding: tuple[int, int] | int = 0,
    stride: tuple[int, int] | int = 1,
    kernel_size: tuple[int, int] = (5, 5),
    epsilon: float | None = None,
    iterations: int = 20,
    local_learning: bool = False,
    local_learning_kl: bool = False,
    momentum: float = 0.1,
    track_running_stats: bool = False,
    type_of_neuron_a: int = 0,
    type_of_neuron_b: int = 0,
    batch_norm_neuron_a: bool = True,
    batch_norm_neuron_b: bool = True,
    bias_norm_neuron_a: bool = False,
    bias_norm_neuron_b: bool = True,
) -> torch.Tensor:

    assert (type_of_neuron_a > 0) or (type_of_neuron_b > 0)

    if number_of_neurons_b <= 0:
        number_of_neurons_b = number_of_neurons_a

    if number_of_neurons_a <= 0:
        number_of_neurons_a = number_of_neurons_b

    assert (type_of_neuron_a == 1) or (type_of_neuron_a == 2)
    assert (type_of_neuron_b == 0) or (type_of_neuron_b == 1) or (type_of_neuron_b == 2) or (type_of_neuron_b == 3)

    kernel_size_internal: list[int] = [kernel_size[-2], kernel_size[-1]]

    if kernel_size[0] < 1:
        kernel_size_internal[0] = test_image.shape[-2]

    if kernel_size[1] < 1:
        kernel_size_internal[1] = test_image.shape[-1]

    network.append(torch.nn.ReLU())
    test_image = network[-1](test_image)

    # I need the output size
    mock_output = (
        torch.nn.functional.conv2d(
            torch.zeros(
                1,
                1,
                test_image.shape[2],
                test_image.shape[3],
            ),
            torch.zeros((1, 1, kernel_size_internal[0], kernel_size_internal[1])),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        .squeeze(0)
        .squeeze(0)
    )
    network.append(
        torch.nn.Unfold(
            kernel_size=(kernel_size_internal[-2], kernel_size_internal[-1]),
            dilation=dilation,
            padding=padding,
            stride=stride,
        )
    )
    test_image = network[-1](test_image)

    network.append(
        torch.nn.Fold(
            output_size=mock_output.shape,
            kernel_size=(1, 1),
            dilation=1,
            padding=0,
            stride=1,
        )
    )
    test_image = network[-1](test_image)

    network.append(L1NormLayer())
    test_image = network[-1](test_image)

    if type_of_neuron_a == 1:
        network.append(
            NNMF2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_neurons_a,
                epsilon=epsilon,
                iterations=iterations,
                local_learning=local_learning,
                local_learning_kl=local_learning_kl,
            ).to(device)
        )
        test_image = network[-1](test_image)
        append_parameter(module=network[-1], parameter_list=parameter_neuron_a)

    elif type_of_neuron_a == 2:
        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_neurons_a,
                kernel_size=(1, 1),
                bias=bias_norm_neuron_a,
            ).to(device)
        )
        test_image = network[-1](test_image)
        append_parameter(module=network[-1], parameter_list=parameter_neuron_a)
    else:
        assert (type_of_neuron_a == 1) or (type_of_neuron_a == 2)

    if batch_norm_neuron_a:
        if (test_image.shape[-1] > 1) or (test_image.shape[-2] > 1):
            network.append(
                torch.nn.BatchNorm2d(
                    num_features=test_image.shape[1],
                    momentum=momentum,
                    track_running_stats=track_running_stats,
                    device=device,
                )
            )
            test_image = network[-1](test_image)
            append_parameter(module=network[-1], parameter_list=parameter_batchnorm2d)

    if type_of_neuron_b == 0:
        pass
    elif type_of_neuron_b == 1:

        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

        network.append(L1NormLayer())
        test_image = network[-1](test_image)

        network.append(
            NNMF2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_neurons_b,
                epsilon=epsilon,
                iterations=iterations,
                local_learning=local_learning,
                local_learning_kl=local_learning_kl,
            ).to(device)
        )
        # Init the cnn top layers 1x1 conv2d layers
        for name, param in network[-1].named_parameters():
            with torch.no_grad():
                print(param.shape)
                if name == "weight":
                    if number_of_neurons_a >= param.shape[0]:
                        param.data[: param.shape[0], : param.shape[0]] = torch.eye(
                            param.shape[0], dtype=param.dtype, device=param.device
                        )
                        param.data[param.shape[0] :, :] = 0
                        param.data[:, param.shape[0] :] = 0
                        param.data += 1.0 / 10000.0

        test_image = network[-1](test_image)
        append_parameter(module=network[-1], parameter_list=parameter_neuron_b)

    elif type_of_neuron_b == 2:

        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

        network.append(L1NormLayer())
        test_image = network[-1](test_image)

        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_neurons_b,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=bias_norm_neuron_b,
                device=device,
            )
        )
        # Init the cnn top layers 1x1 conv2d layers
        for name, param in network[-1].named_parameters():
            with torch.no_grad():
                if name == "bias":
                    param.data *= 0
                    param.data += (torch.rand_like(param) - 0.5) / 10000.0
                if name == "weight":
                    if number_of_neurons_b >= param.shape[0]:
                        assert param.shape[-2] == 1
                        assert param.shape[-1] == 1
                        param.data[: param.shape[0], : param.shape[0], 0, 0] = (
                            torch.eye(
                                param.shape[0], dtype=param.dtype, device=param.device
                            )
                        )
                        param.data[param.shape[0] :, :, 0, 0] = 0
                        param.data[:, param.shape[0] :, 0, 0] = 0
                        param.data += (torch.rand_like(param) - 0.5) / 10000.0

        test_image = network[-1](test_image)
        append_parameter(module=network[-1], parameter_list=parameter_neuron_b)
    
    elif type_of_neuron_b == 3: # W positive
        import torch.nn.utils.parametrize as P

        network.append(torch.nn.ReLU())
        test_image = network[-1](test_image)

        network.append(L1NormLayer())
        test_image = network[-1](test_image)

        network.append(
            torch.nn.Conv2d(
                in_channels=test_image.shape[1],
                out_channels=number_of_neurons_b,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=bias_norm_neuron_b,
                device=device,
            )
        )
        # Init the cnn top layers 1x1 conv2d layers
        for name, param in network[-1].named_parameters():
            with torch.no_grad():
                if name == "bias":
                    param.data *= 0
                    param.data += (torch.rand_like(param) - 0.5) / 10000.0
                if name == "weight":
                    if number_of_neurons_b >= param.shape[0]:
                        assert param.shape[-2] == 1
                        assert param.shape[-1] == 1
                        param.data[: param.shape[0], : param.shape[0], 0, 0] = (
                            torch.eye(
                                param.shape[0], dtype=param.dtype, device=param.device
                            )
                        )
                        param.data[param.shape[0] :, :, 0, 0] = 0
                        param.data[:, param.shape[0] :, 0, 0] = 0
                        param.data += (torch.rand_like(param) - 0.5) / 10000.0

                        param.data = torch.nn.Parameter(torch.abs(param.data))

        class positive_weight(torch.nn.Module):
            def forward(self, x):
                return torch.abs(x)
            
        P.register_parametrization(network[-1], "weight", positive_weight())
        test_image = network[-1](test_image)
        append_parameter(module=network[-1], parameter_list=parameter_neuron_b)
    
    
    else:
        raise ValueError("Unknown type of neuron")
    if (test_image.shape[-1] > 1) or (test_image.shape[-2] > 1):
        if (batch_norm_neuron_b) and (type_of_neuron_b > 0):
            network.append(
                torch.nn.BatchNorm2d(
                    num_features=test_image.shape[1],
                    device=device,
                    momentum=momentum,
                    track_running_stats=track_running_stats,
                )
            )
            test_image = network[-1](test_image)
            append_parameter(module=network[-1], parameter_list=parameter_batchnorm2d)

    return test_image
