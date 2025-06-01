"""Python utility module to support testing.

This is not intended for installation as a package.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from argparse import ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter
import sys
import unittest


class HelpFormatter(ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter):
    """Custom ArgumentParser formatter class.

    This lets us show argument defaults, fix the output width, and also allow
    the raw description formatting that is typically desired.
    """

    # technically an implementation detail
    def __init__(self, prog, indent_increment=2, max_help_position=24, width=80):
        super().__init__(
            prog,
            indent_increment=indent_increment,
            max_help_position=max_help_position,
            width=width
        )


def list_unittest_tests(modname: str, show_active: bool = False) -> list[str]:
    """Returns a list of every unittest test method in the given module.

    This checks every ``unittest.TestCase`` subclass in the module and returns
    a list of <test-case>.<test-method> method names that can be used as
    arguments to ``unittest.main`` either on the command-line or to its
    ``defaultTest`` optional keyword argument.

    If ``show_active`` is provided, each test case name will be prepended by a
    "*" if it is active (not skipped). Skipped tests are prepended by " ".

    .. note::

       You *must* run this function *after* all test case classes have been
       defined in the module or some test cases will not be included.

    Parameters
    ----------
    modname : str
        Name of the module, typically ``__name__``
    show_active : bool, default=False
        Indicate whether or not the skip indicator should be prepended or not.
    """
    # load the module
    mod = sys.modules[modname]
    # find all class names that are subclasses of TestCase
    test_classes = [
        getattr(mod, attr) for attr in dir(mod)
        if isinstance(getattr(mod, attr), type) and
        issubclass(getattr(mod, attr), unittest.TestCase)
    ]

    # helper function to create the class-prefixed test case namem possibly
    # with the skip indicator. this tests for the undocumented
    # __unittest_skip__ property if show_active=True
    def _make_name(cls: type, fname: str) -> str:
        # plain class + function name
        name = f"{cls.__name__}.{fname}"
        # no skip indicator checking
        if not show_active:
            return name
        # otherwise, check if skipped or not
        meth = getattr(cls, fname)
        # note: test classes can be skipped too
        is_skipped = (
            hasattr(meth, "__unittest_skip__") or
            hasattr(cls, "__unittest_skip__")
        )
        return f'{" " if is_skipped else "*"}{name}'

    # build list of class-qualified test case names
    loader = unittest.defaultTestLoader
    return [
        _make_name(test_class, test_name)
        for test_class in test_classes
        for test_name in loader.getTestCaseNames(test_class)
    ]
