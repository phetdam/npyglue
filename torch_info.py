"""Script to print some PyTorch configuration info.

Since PyTorch takes several seconds to load, especially on Windows, we
collect several pieces of configuration information at once:

    * The PyTorch major.minor.patch version
    * The PyTorch flavor, e.g. cpu, xpu, cu124, etc.
    * The PyTorch CMake config script prefix
    * The PyTorch CUDA version if any

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from torch import utils, version


if __name__ == "__main__":
    # split version into major.minor.patch[+flavor]
    ver_cmps = version.__version__.split("+")
    # print major.minor.patch version
    print(ver_cmps[0])
    # print flavor, defaulting to CPU if no flavor
    print("cpu" if len(ver_cmps) == 1 else ver_cmps[-1])
    # CMake prefix to look for TorchConfig.cmake (not the install root)
    print(utils.cmake_prefix_path)
    # CUDA version used to build PyTorch
    print(version.cuda)
