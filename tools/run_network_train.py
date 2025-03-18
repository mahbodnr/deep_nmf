import time
import numpy as np
import torch

import json
from jsmin import jsmin
import os

from torch.utils.tensorboard import SummaryWriter

from tools.make_network import make_network
from tools.get_the_data import get_the_data
from tools.loss_function import loss_function
from tools.make_optimize import make_optimize


def main(
    rand_seed: int = 21,
    only_print_network: bool = False,
    config_network_filename: str = "configs/networks/cnnmf_1x1cnn.json",
    config_data_filename: str = "configs/config_data.json",
    config_lr_parameter_filename: str = "configs/config_lr_parameter.json",
    log_dir: str = None,
) -> None:

    os.makedirs("Models", exist_ok=True)

    device: torch.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    torch.set_default_dtype(torch.float32)

    # Some parameters
    with open(config_data_filename, "r") as file:
        minified = jsmin(file.read())
        config_data = json.loads(minified)

    with open(config_lr_parameter_filename, "r") as file:
        minified = jsmin(file.read())
        config_lr_parameter = json.loads(minified)

    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    if (
        str(config_data["dataset"]) == "MNIST"
        or str(config_data["dataset"]) == "FashionMNIST"
    ):
        input_number_of_channel: int = 1
        input_dim_x: int = 24
        input_dim_y: int = 24
    else:
        input_number_of_channel = 3
        input_dim_x = 28
        input_dim_y = 28

    train_dataloader, test_dataloader, train_processing_chain, test_processing_chain = (
        get_the_data(
            str(config_data["dataset"]),
            int(config_data["batch_size_train"]),
            int(config_data["batch_size_test"]),
            device,
            input_dim_x,
            input_dim_y,
            flip_p=float(config_data["flip_p"]),
            jitter_brightness=float(config_data["jitter_brightness"]),
            jitter_contrast=float(config_data["jitter_contrast"]),
            jitter_saturation=float(config_data["jitter_saturation"]),
            jitter_hue=float(config_data["jitter_hue"]),
            da_auto_mode=bool(config_data["da_auto_mode"]),
        )
    )

    (
        network,
        parameters,
        name_list,
    ) = make_network(
        input_dim_x=input_dim_x,
        input_dim_y=input_dim_y,
        input_number_of_channel=input_number_of_channel,
        device=device,
        config_network_filename=config_network_filename,
    )

    print(network)

    print()
    print("Information about used parameters:")
    number_of_parameter: int = 0
    for i, parameter_list in enumerate(parameters):
        count_parameter: int = 0
        for parameter_element in parameter_list:
            count_parameter += parameter_element.numel()
        print(f"{name_list[i]}: {count_parameter}")
        number_of_parameter += count_parameter
    print(f"total number of parameter: {number_of_parameter}")

    if only_print_network:
        exit()

    (
        optimizers,
        lr_schedulers,
    ) = make_optimize(
        parameters=parameters,
        lr_initial=[
            float(config_lr_parameter["lr_initial_neuron_a"]),
            float(config_lr_parameter["lr_initial_neuron_b"]),
            float(config_lr_parameter["lr_initial_norm"]),
            float(config_lr_parameter["lr_initial_batchnorm2d"]),
        ],
    )
    my_string: str = f"seed_{rand_seed}"
    default_path: str = f"{my_string}"
    log_dir: str = log_dir or f"log_{default_path}"

    tb = SummaryWriter(log_dir=log_dir)

    for epoch_id in range(0, int(config_lr_parameter["number_of_epoch"])):
        print()
        print(f"Epoch: {epoch_id}")
        t_start: float = time.perf_counter()

        train_loss: float = 0.0
        train_correct: int = 0
        train_number: int = 0
        test_correct: int = 0
        test_number: int = 0

        # Switch the network into training mode
        network.train()

        # This runs in total for one epoch split up into mini-batches
        for image, target in train_dataloader:

            # Clean the gradient
            for i in range(0, len(optimizers)):
                if optimizers[i] is not None:
                    optimizers[i].zero_grad()  # type: ignore

            output = network(train_processing_chain(image))

            loss = loss_function(
                h=output,
                labels=target,
                number_of_output_neurons=output.shape[1],
                loss_mode=int(config_lr_parameter["loss_mode"]),
                loss_coeffs_mse=float(config_lr_parameter["loss_coeffs_mse"]),
                loss_coeffs_kldiv=float(config_lr_parameter["loss_coeffs_kldiv"]),
            )

            assert loss is not None
            train_loss += loss.item()
            train_correct += (output.argmax(dim=1) == target).sum().cpu().numpy()
            train_number += target.shape[0]

            # Calculate backprop
            loss.backward()

            # Update the parameter
            # Clean the gradient
            for i in range(0, len(optimizers)):
                if optimizers[i] is not None:
                    optimizers[i].step()  # type: ignore

        perfomance_train_correct: float = 100.0 * train_correct / train_number
        # Update the learning rate
        for i in range(0, len(lr_schedulers)):
            if lr_schedulers[i] is not None:
                lr_schedulers[i].step(train_loss)  # type: ignore

        my_string = "Actual lr: "
        for i in range(0, len(lr_schedulers)):
            if lr_schedulers[i] is not None:
                my_string += f" {lr_schedulers[i].get_last_lr()[0]:.4e} "  # type: ignore
            else:
                my_string += " --- "

        print(my_string)
        t_training: float = time.perf_counter()

        # Switch the network into evalution mode
        network.eval()

        with torch.no_grad():

            for image, target in test_dataloader:
                output = network(test_processing_chain(image))

                test_correct += (output.argmax(dim=1) == target).sum().cpu().numpy()
                test_number += target.shape[0]

        t_testing = time.perf_counter()

        perfomance_test_correct: float = 100.0 * test_correct / test_number

        tb.add_scalar("Train Loss", train_loss / float(train_number), epoch_id)
        tb.add_scalar("Train Number Correct", train_correct, epoch_id)
        tb.add_scalar("Test Number Correct", test_correct, epoch_id)

        print(
            f"Training: Loss={train_loss / float(train_number):.5f} Correct={perfomance_train_correct:.2f}%"
        )
        print(f"Testing: Correct={perfomance_test_correct:.2f}%")
        print(
            f"Time: Training={(t_training - t_start):.1f}sec, Testing={(t_testing - t_training):.1f}sec"
        )

        tb.flush()

        lr_check: list[float] = []
        for i in range(0, len(lr_schedulers)):
            if lr_schedulers[i] is not None:
                lr_check.append(lr_schedulers[i].get_last_lr()[0])  # type: ignore

        lr_check_max = float(torch.tensor(lr_check).max())

        if lr_check_max < float(config_lr_parameter["lr_limit"]):
            torch.save(network, f"Models/Model_{default_path}.pt")
            tb.close()
            print("Done (lr_limit)")
            return

    torch.save(network, f"Models/Model_{default_path}.pt")
    print()

    tb.close()
    print("Done (loop end)")

    return
