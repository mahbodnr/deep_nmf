import torch


def append_parameter(
    module: torch.nn.Module, parameter_list: list[torch.nn.parameter.Parameter]
):
    for netp in module.parameters():
        parameter_list.append(netp)
