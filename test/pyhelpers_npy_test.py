"""Python test script for the pyhelpers C++ extension module.

This tests only the pyhelpers features that *do* need NumPy.

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
from pyhelpers_test import _with_arma, _with_eigen

# unittest skip decorators for Eigen3 and Armadillo
_no_eigen3 = "pyhelpers not built with Eigen3"
_skip_if_no_eigen3 = unittest.skipUnless(_with_eigen, _no_eigen3)
_no_arma = "pyhelpers not built with Armadillo"
_skip_if_no_arma = unittest.skipUnless(_with_arma, _no_arma)


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

    @_skip_if_no_eigen3
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

    @_skip_if_no_arma
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

    @_skip_if_no_arma
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

    @_skip_if_no_arma
    def test_arma_rowvec(self):
        """Test created a 1D NumPy array from an arma::rowvec."""
        ar = np.array([4., 2.333, 1.23, 6.45, 28.77, 4.23, 1.115, 12.4])
        cap = ph.make_capsule(ph.CAPSULE_ARMADILLO_ROWVEC)
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
    print(f"Running pyhelpers NumPy tests")
    res = unittest.main(argv=(sys.argv[0],), exit=False, verbosity=1 + argn.verbose)
    return 0 if res.result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
