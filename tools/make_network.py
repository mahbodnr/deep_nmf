import torch
from tools.append_block import append_block
from tools.L1NormLayer import L1NormLayer
from tools.NNMF2d import NNMF2d
from tools.append_parameter import append_parameter

import json
from jsmin import jsmin


def make_network(
    input_dim_x: int,
    input_dim_y: int,
    input_number_of_channel: int,
    device: torch.device,
    config_network_filename: str = "config_network.json",
) -> tuple[
    torch.nn.Sequential,
    list[list[torch.nn.parameter.Parameter]],
    list[str],
]:

    with open(config_network_filename, "r") as file:
        minified = jsmin(file.read())
        config_network = json.loads(minified)

    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["number_of_neurons_b"])
    )

    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["kernel_size_conv"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["stride_conv"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["padding_conv"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["dilation_conv"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["kernel_size_pool"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["stride_pool"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["padding_pool"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["dilation_pool"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["type_of_pooling"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["local_learning_pooling"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["local_learning_use_kl_pooling"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["type_of_neuron_a"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["type_of_neuron_b"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["batch_norm_neuron_a"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["batch_norm_neuron_b"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["bias_norm_neuron_a"])
    )
    assert len(list(config_network["number_of_neurons_a"])) == len(
        list(config_network["bias_norm_neuron_b"])
    )

    parameter_neuron_b: list[torch.nn.parameter.Parameter] = []
    parameter_neuron_a: list[torch.nn.parameter.Parameter] = []
    parameter_batchnorm2d: list[torch.nn.parameter.Parameter] = []
    parameter_neuron_pool: list[torch.nn.parameter.Parameter] = []

    test_image = torch.ones(
        (1, input_number_of_channel, input_dim_x, input_dim_y), device=device
    )

    network = torch.nn.Sequential()
    network = network.to(device)

    epsilon: float | None = None

    if isinstance(config_network["epsilon"], float):
        epsilon = float(config_network["epsilon"])

    for block_id in range(0, len(list(config_network["number_of_neurons_a"]))):

        test_image = append_block(
            network=network,
            number_of_neurons_a=int(
                list(config_network["number_of_neurons_a"])[block_id]
            ),
            number_of_neurons_b=int(
                list(config_network["number_of_neurons_b"])[block_id]
            ),
            test_image=test_image,
            dilation=list(list(config_network["dilation_conv"])[block_id]),
            padding=list(list(config_network["padding_conv"])[block_id]),
            stride=list(list(config_network["stride_conv"])[block_id]),
            kernel_size=list(list(config_network["kernel_size_conv"])[block_id]),
            epsilon=epsilon,
            local_learning = bool(
                list(config_network["local_learning"])[block_id]
            ),
            local_learning_kl = bool(
                list(config_network["local_learning_kl"])[block_id]
            ),
            iterations=int(config_network["iterations"]),
            device=device,
            parameter_neuron_a=parameter_neuron_a,
            parameter_neuron_b=parameter_neuron_b,
            parameter_batchnorm2d=parameter_batchnorm2d,
            type_of_neuron_a=int(list(config_network["type_of_neuron_a"])[block_id]),
            type_of_neuron_b=int(list(config_network["type_of_neuron_b"])[block_id]),
            batch_norm_neuron_a=bool(
                list(config_network["batch_norm_neuron_a"])[block_id]
            ),
            batch_norm_neuron_b=bool(
                list(config_network["batch_norm_neuron_b"])[block_id]
            ),
            bias_norm_neuron_a=bool(
                list(config_network["bias_norm_neuron_a"])[block_id]
            ),
            bias_norm_neuron_b=bool(
                list(config_network["bias_norm_neuron_b"])[block_id]
            ),
        )

        if (int(list(list(config_network["kernel_size_pool"])[block_id])[0]) > 0) and (
            (int(list(list(config_network["kernel_size_pool"])[block_id])[1]) > 0)
        ):
            if int(list(config_network["type_of_pooling"])[block_id]) == 0:
                pass

            elif int(list(config_network["type_of_pooling"])[block_id]) == 1:
                network.append(
                    torch.nn.AvgPool2d(
                        kernel_size=(
                            (
                                int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[0]
                                )
                            ),
                            (
                                int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[1]
                                )
                            ),
                        ),
                        stride=(
                            (
                                int(
                                    list(list(config_network["stride_pool"])[block_id])[
                                        0
                                    ]
                                )
                            ),
                            (
                                int(
                                    list(list(config_network["stride_pool"])[block_id])[
                                        1
                                    ]
                                )
                            ),
                        ),
                        padding=(
                            (
                                int(
                                    list(
                                        list(config_network["padding_pool"])[block_id]
                                    )[0]
                                )
                            ),
                            (
                                int(
                                    list(
                                        list(config_network["padding_pool"])[block_id]
                                    )[1]
                                )
                            ),
                        ),
                    )
                )
                test_image = network[-1](test_image)

            elif int(list(config_network["type_of_pooling"])[block_id]) == 2:
                network.append(
                    torch.nn.MaxPool2d(
                        kernel_size=(
                            (
                                int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[0]
                                )
                            ),
                            (
                                int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[1]
                                )
                            ),
                        ),
                        stride=(
                            (
                                int(
                                    list(list(config_network["stride_pool"])[block_id])[
                                        0
                                    ]
                                )
                            ),
                            (
                                int(
                                    list(list(config_network["stride_pool"])[block_id])[
                                        1
                                    ]
                                )
                            ),
                        ),
                        padding=(
                            (
                                int(
                                    list(
                                        list(config_network["padding_pool"])[block_id]
                                    )[0]
                                )
                            ),
                            (
                                int(
                                    list(
                                        list(config_network["padding_pool"])[block_id]
                                    )[1]
                                )
                            ),
                        ),
                    )
                )
                test_image = network[-1](test_image)
            elif (int(list(config_network["type_of_pooling"])[block_id]) == 3) or (
                int(list(config_network["type_of_pooling"])[block_id]) == 4
            ):

                network.append(torch.nn.ReLU())
                test_image = network[-1](test_image)

                mock_output = (
                    torch.nn.functional.conv2d(
                        torch.zeros(
                            1,
                            1,
                            test_image.shape[2],
                            test_image.shape[3],
                        ),
                        torch.zeros(
                            (
                                1,
                                1,
                                int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[0]
                                ),
                                int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[1]
                                ),
                            )
                        ),
                        stride=(
                            (
                                int(
                                    list(list(config_network["stride_pool"])[block_id])[
                                        0
                                    ]
                                )
                            ),
                            (
                                int(
                                    list(list(config_network["stride_pool"])[block_id])[
                                        1
                                    ]
                                )
                            ),
                        ),
                        padding=(
                            (
                                int(
                                    list(
                                        list(config_network["padding_pool"])[block_id]
                                    )[0]
                                )
                            ),
                            (
                                int(
                                    list(
                                        list(config_network["padding_pool"])[block_id]
                                    )[1]
                                )
                            ),
                        ),
                        dilation=(
                            (
                                int(
                                    list(
                                        list(config_network["dilation_pool"])[block_id]
                                    )[0]
                                )
                            ),
                            (
                                int(
                                    list(
                                        list(config_network["dilation_pool"])[block_id]
                                    )[1]
                                )
                            ),
                        ),
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

                network.append(
                    torch.nn.Unfold(
                        kernel_size=(
                            int(
                                list(
                                    list(config_network["kernel_size_pool"])[block_id]
                                )[0]
                            ),
                            int(
                                list(
                                    list(config_network["kernel_size_pool"])[block_id]
                                )[1]
                            ),
                        ),
                        stride=(
                            (
                                int(
                                    list(list(config_network["stride_pool"])[block_id])[
                                        0
                                    ]
                                )
                            ),
                            (
                                int(
                                    list(list(config_network["stride_pool"])[block_id])[
                                        1
                                    ]
                                )
                            ),
                        ),
                        padding=(
                            (
                                int(
                                    list(
                                        list(config_network["padding_pool"])[block_id]
                                    )[0]
                                )
                            ),
                            (
                                int(
                                    list(
                                        list(config_network["padding_pool"])[block_id]
                                    )[1]
                                )
                            ),
                        ),
                        dilation=(
                            (
                                int(
                                    list(
                                        list(config_network["dilation_pool"])[block_id]
                                    )[0]
                                )
                            ),
                            (
                                int(
                                    list(
                                        list(config_network["dilation_pool"])[block_id]
                                    )[1]
                                )
                            ),
                        ),
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

                if int(list(config_network["type_of_pooling"])[block_id]) == 3:
                    network.append(
                        torch.nn.Conv2d(
                            in_channels=test_image.shape[1],
                            out_channels=test_image.shape[1]
                            // (
                                int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[0]
                                )
                                * int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[1]
                                )
                            ),
                            kernel_size=(1, 1),
                            bias=False,
                        ).to(device)
                    )
                else:
                    network.append(
                        NNMF2d(
                            in_channels=test_image.shape[1],
                            out_channels=test_image.shape[1]
                            // (
                                int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[0]
                                )
                                * int(
                                    list(
                                        list(config_network["kernel_size_pool"])[
                                            block_id
                                        ]
                                    )[1]
                                )
                            ),
                            epsilon=epsilon,
                            local_learning=bool(
                                list(config_network["local_learning_pooling"])[block_id]
                            ),
                            local_learning_kl=bool(
                                list(config_network["local_learning_use_kl_pooling"])[
                                    block_id
                                ]
                            ),
                        ).to(device)
                    )

                test_image = network[-1](test_image)
                append_parameter(
                    module=network[-1], parameter_list=parameter_neuron_pool
                )

                network.append(
                    torch.nn.BatchNorm2d(
                        num_features=test_image.shape[1],
                        device=device,
                        momentum=0.1,
                        track_running_stats=False,
                    )
                )
                test_image = network[-1](test_image)
                append_parameter(
                    module=network[-1], parameter_list=parameter_batchnorm2d
                )

            else:
                assert int(list(config_network["type_of_pooling"])[block_id]) > 4
    network.append(torch.nn.Softmax(dim=1))
    test_image = network[-1](test_image)

    network.append(torch.nn.Flatten())
    test_image = network[-1](test_image)

    parameters: list[list[torch.nn.parameter.Parameter]] = [
        parameter_neuron_a,
        parameter_neuron_b,
        parameter_batchnorm2d,
        parameter_neuron_pool,
    ]

    name_list: list[str] = ["neuron a", "neuron b", "batchnorm2d", "neuron pool"]

    return (
        network,
        parameters,
        name_list,
    )
