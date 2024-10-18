"""Script to print some PyTorch configuration info.

Since PyTorch takes several seconds to load we collect the version, whether or
not acceleration via CUDA is available, and the path to the CMake modules.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from torch import __version__, cuda, utils


if __name__ == "__main__":
    print(__version__)
    print(utils.cmake_prefix_path)
    print(cuda.is_available())
