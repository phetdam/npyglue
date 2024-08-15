"""Python test script for the pymath C++ extension modules.

Provides command-line options for controlling what pymath extension module
variant is loaded, e.g. hand vs. SWIG wrapped, default C++ vs. C++20 standard.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    RawDescriptionHelpFormatter
)
import importlib
import os
from os.path import abspath, dirname, relpath
import sys
from typing import Iterable, Optional
import unittest

import numpy as np
from numpy.testing import assert_allclose

# current working directory + file directory
_cwd = os.getcwd()
_home_dir = dirname(abspath(__file__))

# add current working directory to import path
sys.path.insert(0, os.getcwd())

# module handle. this is set by main() after looking at CLI options
pm = None


class HelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """Custom ArgumentParser formatter class."""

    # technically an implementation detail
    def __init__(self, prog, indent_increment=2, max_help_position=24, width=80):
        super().__init__(
            prog,
            indent_increment=indent_increment,
            max_help_position=max_help_position,
            width=width
        )


class TestArrayDouble(unittest.TestCase):
    """Test suite for array_double tests."""

    def test_double_list(self):
        """Test array_double on a nested list."""
        in_list = [[[3, 4], [2, 4]], [[1, 3], [5, 2]]]
        assert_allclose(pm.array_double(in_list), 2 * np.array(in_list))

    def test_fdouble_list(self):
        """Test farray_double on a nested list."""
        in_list = [[[1, 4.4]], [[3.2, 15]], [[4, 2.2222]]]
        assert_allclose(pm.farray_double(in_list), 2 * np.array(in_list))

    def test_double_array(self):
        """Test array_double on a NumPy array."""
        in_ar = np.array([3., 4.2, 1.222, 1.2])
        assert_allclose(pm.array_double(in_ar), 2 * in_ar)

    def test_fdouble_array(self):
        """Test farray_double on a NumPy array."""
        # cannot safely convert from float64 -> float32 so specify dtype
        in_ar = np.array([[2.3, 1.111], [4.3, 4.111]], dtype=np.float32)
        assert_allclose(pm.farray_double(in_ar), 2 * in_ar)


class TestUnitCompress(unittest.TestCase):
    """Test suite for unit_compress tests."""

    def test_compress_list(self):
        """Test unit_compress on a nested list."""
        in_list = [[3.4, 1.1111], [4, 3]]
        assert_allclose(
            pm.unit_compress(in_list),
            np.array(in_list) / np.max(in_list)
        )

    def test_fcompress_list(self):
        """Test funit_compress on a nested list."""
        in_list = [[[2]], [[2.333]], [[1.111]], [[12]]]
        assert_allclose(
            pm.funit_compress(in_list),
            np.array(in_list) / np.max(in_list)
        )

    def test_compress_array(self):
        """Test unit_compress on a NumPy array."""
        in_ar = np.array([3., 2.111, 1.2131, 1.563, 3.4522])
        assert_allclose(pm.unit_compress(in_ar), in_ar / np.max(in_ar))

    def test_fcompress_array(self):
        """Test ufnit_compress on a NumPy array."""
        in_ar = np.array([[[4.33]], [[3.22343]], [[3.11]]], dtype=np.float32)
        assert_allclose(pm.funit_compress(in_ar), in_ar / np.max(in_ar))


def main(args: Optional[Iterable[str]] = None) -> int:
    """Main function.

    Parameters
    ----------
    args : Iterable[str], default=None
        Command-line arguments, if ``None`` uses ``sys.argv``

    Returns
    -------
    int
        Exit status, 0 for success
    """
    # parse CLI arguments
    ap = ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    ap.add_argument(
        "-f",
        "--flavor",
        choices=("hand", "swig"),
        default="hand",
        help="C++ function wrapping method"
    )
    ap.add_argument(
        "-std",
        "--cc-standard",
        choices=("cc", "cc20"),
        default="cc",
        help="C++ standard used during compilation"
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Run more verbosely"
    )
    argn = ap.parse_args(args=args)
    # determine name of the pymath library we are going to load
    mod_name = "pymath"
    if argn.flavor == "swig":
        mod_name += "_swig"
    if argn.cc_standard != "cc":
        mod_name += f"_{argn.cc_standard}"
    # indicate what will be loaded
    if argn.verbose:
        print(f"loading {mod_name} from {relpath(_cwd, _home_dir)}...")
    # attempt to import
    global pm
    pm = importlib.import_module(mod_name)
    if argn.verbose:
        print(f"loading {mod_name} from {relpath(_cwd, _home_dir)}... done")
    # run tests. trick unittest.main into thinking there are no CLI args
    res = unittest.main(argv=(sys.argv[0],), exit=False, verbosity=1 + argn.verbose)
    return 0 if res.result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
