import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argh

import numpy as np
import torch

rand_seed: int = 21
torch.manual_seed(rand_seed)
torch.cuda.manual_seed(rand_seed)
np.random.seed(rand_seed)

from get_the_data_uniform import get_the_data


def main(
    dataset: str = "CIFAR10",  # "CIFAR10", "FashionMNIST", "MNIST"
    only_print_network: bool = False,
    model_name: str = "Model_iter20_lr_1.0000e-03_1.0000e-02_1.0000e-03_.pt",
) -> None:

    torch_device: torch.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    torch.set_default_dtype(torch.float32)

    # Some parameters
    batch_size_test: int = 50  # 0

    loss_mode: int = 0
    loss_coeffs_mse: float = 0.5
    loss_coeffs_kldiv: float = 1.0
    print(
        "loss_mode: ",
        loss_mode,
        "loss_coeffs_mse: ",
        loss_coeffs_mse,
        "loss_coeffs_kldiv: ",
        loss_coeffs_kldiv,
    )

    if dataset == "MNIST" or dataset == "FashionMNIST":
        input_dim_x: int = 24
        input_dim_y: int = 24
    else:
        input_dim_x = 28
        input_dim_y = 28

    test_dataloader, test_processing_chain = get_the_data(
        dataset,
        batch_size_test,
        torch_device,
        input_dim_x,
        input_dim_y,
    )

    network = torch.load(model_name)
    network.to(device=torch_device)

    print(network)

    if only_print_network:
        exit()

    # Switch the network into evalution mode
    network.eval()
    number_of_noise_steps = 20
    noise_scale = torch.arange(0, number_of_noise_steps + 1) / float(
        number_of_noise_steps
    )

    results = torch.zeros_like(noise_scale)

    with torch.no_grad():

        for position in range(0, noise_scale.shape[0]):
            test_correct: int = 0
            test_number: int = 0
            eta: float = noise_scale[position]
            for image, target in test_dataloader:
                noise = torch.rand_like(image) > eta

                image = image * noise
                image = image / (image.sum(dim=(1, 2, 3), keepdim=True) + 1e-20)
                output = network(test_processing_chain(image))

                test_correct += (output.argmax(dim=1) == target).sum().cpu().numpy()
                test_number += target.shape[0]

            perfomance_test_correct: float = 100.0 * test_correct / test_number
            results[position] = perfomance_test_correct

            print(f"{eta:.2f}: {perfomance_test_correct:.2f}%")

    np.save("noise_holes_results.npy", results.cpu().numpy())
    return


if __name__ == "__main__":
    argh.dispatch_command(main)
