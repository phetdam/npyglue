"""Script to print some PyTorch configuration info.

Since PyTorch takes several seconds to load we collect the version, whether or
not acceleration via CUDA is available, and the path to the CMake modules.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from torch import utils, version


if __name__ == "__main__":
    print(version.__version__)
    print(utils.cmake_prefix_path)
    print(version.cuda)
