"""Python test script for the pyhelpers C++ extension module.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from argparse import ArgumentParser
import os
import sys
from typing import Iterable, Optional
import unittest

import numpy as np
from numpy.testing import assert_allclose

# add working directory + root directory to import paths
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from npygl_utils import HelpFormatter

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


class TestArrayFromCapsule(unittest.TestCase):
    """Test suite for array_from_capsule."""

    def test_std_map(self):
        """Test creating a NumPy array from a std::map capsule.

        This is expected to fail since we allow numeric objects only.
        """
        with self.assertRaises(TypeError):
            ph.array_from_capsule(ph.make_capsule(ph.CAPSULE_STD_MAP))

    def test_std_vector(self):
        """Test creating a 1D NumPy array from a std::vector<double>."""
        ar = np.array([4.3, 3.222, 4.12, 1.233, 1.66, 6.55, 77.333])
        cap = ph.make_capsule(ph.CAPSULE_STD_VECTOR)
        assert_allclose(ar, ph.array_from_capsule(cap))

    @unittest.skipUnless(_with_eigen, "pyhelpers not built with Eigen3")
    def test_eigen3_matrix(self):
        """Test creating a 2D NumPy array from a Eigen::MatrixXf."""
        ar = np.array(
            [
                [3.4, 1.222, 5.122, 1.22],
                [6.44, 3.21, 5.345, 9.66],
                [6.244, 3.414, 1.231, 4.85]
            ],
            dtype=np.float32
        )
        cap = ph.make_capsule(ph.CAPSULE_EIGEN3_MATRIX)
        assert_allclose(ph.array_from_capsule(cap), ar)

    @unittest.skipUnless(_with_arma, "pyhelpers not built with Armadillo")
    def test_arma_matrix(self):
        """Test creating a 2D NumPy array from an arma::fmat."""
        ar = np.array(
            [
                [1., 2.33, 4.33, 1.564],
                [4.55, 55.6, 1.212, 4.333],
                [4.232, 9.83, 1.56, 65.34]
            ],
            dtype=np.float32
        )
        cap = ph.make_capsule(ph.CAPSULE_ARMADILLO_MATRIX)
        assert_allclose(ph.array_from_capsule(cap), ar)

    @unittest.skipUnless(_with_arma, "pyhelpers not built with Armadillo")
    def test_arma_cube(self):
        """Test creating a 3D NumPy array from an arma::cx_cube."""
        # (n_rows, n_cols, n_slices) as Armadillo cube is not a tensor
        ar = np.array(
            [
                # row 1
                [
                    # columns 1, 2, 3
                    [4.33+2.323j, 5.66+5.11j],      # slices 1, 2
                    [4.23+0.222j, 5.66+11.222j],    # slices 1, 2
                    [6.345+1.1j, 5.44+22.333j]      # slices 1, 2
                ],
                # row 2
                [
                    # columns 1, 2, 3
                    [6.55+4.12j, 6.5+1.22j],        # slices 1, 2
                    [4.11+1.2323j, 1.222+11.888j],  # slices 1, 2
                    [6.77+9.88j, 9.88+2.33j]        # slices 1, 2
                ]
            ]
        )
        cap = ph.make_capsule(ph.CAPSULE_ARMADILLO_CUBE)
        assert_allclose(ph.array_from_capsule(cap), ar)


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
