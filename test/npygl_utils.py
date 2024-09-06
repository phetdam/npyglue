"""Python utility module to support testing.

This is not intended for installation as a package.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from argparse import ArgumentDefaultsHelpFormatter, RawDescriptionHelpFormatter


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
