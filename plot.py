import numpy as np
import matplotlib.pyplot as plt

data = np.load("data_log.npy")
plt.loglog(
    data[:, 0],
    100.0 * (1.0 - data[:, 1] / 10000.0),
    "k",
)

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error [%]")
plt.title("CIFAR10")
plt.show()
