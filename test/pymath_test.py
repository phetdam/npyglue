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
import sys
from typing import Iterable, Optional
import unittest

import numpy as np
from numpy.testing import assert_allclose

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


class TestSine(unittest.TestCase):
    """Test suite for sine tests."""

    def test_sine_list(self):
        """Test sine on a nested list."""
        in_list = [[[4.3]], [[1.3343]], [[12]]]
        assert_allclose(pm.sine(in_list), np.sin(in_list))

    def test_fsine_list(self):
        """Test fsine on a nested list."""
        in_list = [[3.4, 2], [1.22, 4.5], [1.33, 1]]
        assert_allclose(pm.fsine(in_list), np.sin(in_list, dtype=np.float32))

    def test_sine_array(self):
        """Test sine on a NumPy array."""
        in_ar = np.array([[[3.444]], [[2.33]], [[1.2121]]])
        assert_allclose(pm.sine(in_ar), np.sin(in_ar))

    def test_fsine_array(self):
        """Test fsine on a NumPy array."""
        in_ar = np.array([[3.4, 2.11, 1.22], [2.323, 1.11, 1.141]], dtype=np.float32)
        assert_allclose(pm.fsine(in_ar), np.sin(in_ar))


class TestNorm1(unittest.TestCase):
    """Test suite for norm1 tests."""

    def test_norm1_list(self):
        """Test norm1 on a flat list."""
        in_list = [1., 1.3, 1.222, 1.452, 6.55]
        assert_allclose(pm.norm1(in_list), np.abs(in_list).sum())

    def test_fnorm1_list(self):
        """Test fnorm1 on a nested list."""
        in_list = [[2.], [1.334], [2.4322], [5.44431]]
        assert_allclose(pm.fnorm1(in_list), np.abs(in_list, dtype=np.float32).sum())

    def test_norm1_tuple(self):
        """Test norm1 on a tuple with nested lists."""
        in_tup = ([(3., 2.)], [(4.33, 1.444)], ([1.342, 9.827],))
        assert_allclose(pm.norm1(in_tup), np.abs(in_tup).sum())

    def test_fnorm1_tuple(self):
        """Test fnorm1 on a tuple with nested tuples."""
        in_tup = (((1,),), ((3.2222,),), ((1.42423,),))
        assert_allclose(pm.fnorm1(in_tup), np.abs(in_tup, dtype=np.float32).sum())

    def test_norm1_array(self):
        """Test norm1 on a NumPy array."""
        in_ar = np.array([[[3., 1.212]], [[1.222, 1.2123]]])
        assert_allclose(pm.norm1(in_ar), np.abs(in_ar).sum())

    def test_fnorm1_array(self):
        """Test fnorm1 on a flat NumPy array."""
        in_ar = np.array([3., 1.212, 1.222, 1.22123], dtype=np.float32)
        assert_allclose(pm.fnorm1(in_ar), np.abs(in_ar, dtype=np.float32).sum())


class TestNorm2(unittest.TestCase):
    """Test suite for norm2 tests."""

    def test_norm2_list(self):
        """Test norm2 on a nested list."""
        in_list = [[1., 2.3232], [1.444, 5.3232], [5.6666, 4.222]]
        assert_allclose(pm.norm2(in_list), np.linalg.norm(in_list))

    def test_fnorm2_list(self):
        """Test fnorm2 on a nested list."""
        in_list = [[[3.43434]], [[1.12121]], [[4.33]], [[7.545]]]
        assert_allclose(
            pm.fnorm2(in_list),
            np.linalg.norm(np.array(in_list, dtype=np.float32))
        )

    def test_norm2_array(self):
        """Test norm2 on a NumPy array."""
        in_ar = np.array([1.3, 23.22, 1.44, 1.32, 1.2323, 6.55, 34.333])
        assert_allclose(pm.norm2(in_ar), np.linalg.norm(in_ar))

    def test_fnorm2_array(self):
        """Test fnorm2 on a NumPy array."""
        in_ar = np.array([[[[4.3]]], [[[8.333]]], [[[5.612]]]], dtype=np.float32)
        assert_allclose(
            pm.fnorm2(in_ar),
            np.linalg.norm(np.array(in_ar, dtype=np.float32))
        )


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
    if argn.flavor != "hand":
        mod_name += f"_{argn.flavor}"
    if argn.cc_standard != "cc":
        mod_name += f"_{argn.cc_standard}"
    # indicate what will be loaded
    if argn.verbose:
        print(f"loading {mod_name}...")
    # attempt to import
    global pm
    pm = importlib.import_module(mod_name)
    if argn.verbose:
        print(f"loading {mod_name}... done")
    # run tests. trick unittest.main into thinking there are no CLI args
    res = unittest.main(argv=(sys.argv[0],), exit=False, verbosity=1 + argn.verbose)
    return 0 if res.result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
