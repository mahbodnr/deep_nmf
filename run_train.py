import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argh
from tools.run_network_train import main

if __name__ == "__main__":
    argh.dispatch_command(main)