"""Python test script for the pyhelpers C++ extension module.

This tests only the pyhelpers features that do not need NumPy.

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

from npygl_utils import (
    HelpFormatter,
    add_test_filtering_options,
    list_unittest_tests
)

import pyhelpers as ph  # type: ignore

# indicate if module was built with Eigen or not
_with_eigen = True if ph.eigen3_version() else False
# indicate if module was built with Armadillo or not
_with_arma = True if ph.armadillo_version() else False
# constants for the different capsule creation methods
_cap_methods = [(a, getattr(ph, a)) for a in dir(ph) if a.startswith("CAPSULE_")]


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


class TestArmadilloVersion(unittest.TestCase):
    """Test suite for armadillo_version."""

    def test(self):
        """Basic functional test."""
        if _with_arma:
            ver = ph.armadillo_version()
            self.assertEqual(str, type(ver))
        else:
            self.assertIsNone(ph.armadillo_version())


class TestCapsuleMap(unittest.TestCase):
    """Test suite for capsule_map."""

    def test(self):
        """Basic functional test."""
        self.assertEqual("PyCapsule", type(ph.capsule_map()).__name__)


class TestMakeCapsule(unittest.TestCase):
    """Test suite for make_capsule."""

    def test(self):
        """Basic functional test.

        We want to be able to create a capsule and then get its string type.
        """
        for m_name, m_value in _cap_methods:
            with self.subTest(m_name=m_name, m_value=m_value):
                cap = ph.make_capsule(m_value)
                self.assertEqual(str, type(ph.capsule_type(cap)))


class TestCapsuleStr(unittest.TestCase):
    """Test suite for capsule_str."""

    def test(self):
        """Basic functional test.

        We want to create a capsule and then get is string representation. All
        the available capsule types should be supported.
        """
        for m_name, m_value in _cap_methods:
            with self.subTest(m_name=m_name, m_value=m_value):
                cap = ph.make_capsule(m_value)
                self.assertEqual(str, type(ph.capsule_str(cap)))


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
    add_test_filtering_options(ap)
    argn = ap.parse_args(args=args)
    # list test cases if requested
    if argn.list_tests:
        for test_name in list_unittest_tests(__name__):
            print(test_name)
        return 0
    # run tests. trick unittest.main into thinking there are no CLI args
    print(f"Running pyhelpers tests")
    res = unittest.main(
        defaultTest=argn.tests,
        argv=(sys.argv[0],),
        exit=False,
        verbosity=1 + argn.verbose
    )
    return 0 if res.result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
