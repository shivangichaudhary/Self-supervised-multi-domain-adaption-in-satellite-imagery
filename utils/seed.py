import random
import numpy as np
import torch


def set_seed(seed: int):
    """Seeds python, numpy and torch (CPU + CUDA) so a training run is repeatable.

    cuDNN autotuning is left on for speed, so GPU runs are repeatable to within
    floating-point nondeterminism of the convolution kernels, not bit-exact.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
