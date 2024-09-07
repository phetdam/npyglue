"""Python test script for the pyhelpers C++ extension module.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from argparse import ArgumentParser
import os
import sys
from typing import Iterable, Optional
import unittest

# add working directory + root directory to import paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npygl_utils import HelpFormatter

import pyhelpers as ph  # type: ignore

# indicate if module was built with Eigen or not
_with_eigen = hasattr(ph, "CAPSULE_EIGEN3_MATRIX")


class TestParseArgs1(unittest.TestCase):
    """Test suite for parse_args_1."""

    def test_string(self):
        """Test with a string."""
        s = "my_string"
        self.assertEqual(repr(type(s)), ph.parse_args_1(s))

    def test_list(self):
        """Test with a list."""
        o = [4., "asdf", [4., 111]]
        self.assertEqual(repr(type(o)), ph.parse_args_1(o))

    def test_dict(self):
        """Test with a dict."""
        d = {"a": 3, "b": [4, 5, 12, 11]}
        self.assertEqual(repr(type(d)), ph.parse_args_1(d))

    def test_arg_check(self):
        """Test internal argument count check."""
        with self.assertRaises(TypeError):
            ph.parse_args_1(2, complex(3.22, 1.))


class TestEigen3Version(unittest.TestCase):
    """Test suite for eigen3_version."""

    def test(self):
        """Basic functional test."""
        if _with_eigen:
            ver = ph.eigen3_version()
            self.assertEqual(str, type(ver))
            # should have world version 3
            self.assertEqual("3", ver.split(".")[0])
        else:
            self.assertIsNone(ph.eigen3_version())


class TestCapsuleMap(unittest.TestCase):
    """Test suite for capsule_map."""

    def test(self):
        """Basic functional test."""
        self.assertEqual("PyCapsule", type(ph.capsule_map()).__name__)


def main(args: Optional[Iterable[str]] = None) -> int:
    """Main function for the testing script.

    Parameters
    ----------
    args : Iterable[str], default=None
        Command-line arguments for argparse to parse

    Returns
    -------
    int
        Exit code
    """
    # parse arguments
    ap = ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Run more verbosely"
    )
    argn = ap.parse_args(args=args)
    # run tests. trick unittest.main into thinking there are no CLI args
    print(f"Running pyhelpers tests")
    res = unittest.main(argv=(sys.argv[0],), exit=False, verbosity=1 + argn.verbose)
    return 0 if res.result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
