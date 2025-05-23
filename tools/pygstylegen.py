"""Generate the CSS stylesheet for the specified highlighting style.

This stylesheet will be used to control CSS styling of the HTML injected into
Markdown files consumed by Doxygen that are produced by pyginject.py.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>
"""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
import sys
from typing import Iterable, Optional

from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name

# default Pygments style we like
_default_style = "one-dark"

# argument parser help
_help_style = f"Pygments style to use, default {_default_style}"
_help_output_file = "CSS output file. If not provided, stdout is used"


def parse_args(args: Optional[Iterable[str]] = None) -> Namespace:
    """Parse incoming arguments.

    Parameters
    ----------
    args : Iterable[str], default=None
        Iterable of command-line arguments

    Returns
    -------
    Namespace
    """
    # argument parser
    ap = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter
    )
    # style + output file
    ap.add_argument(
        "-s",
        "--style",
        type=get_style_by_name,
        default=_default_style,
        help=_help_style,
    )
    ap.add_argument("-o", "--output-file", help=_help_output_file)
    # parse args and return
    return ap.parse_args(args=args)


def main(args: Optional[Iterable[str]] = None) -> int:
    """Main function.

    Parameters
    ----------
    args : Iterable[str], default=None
        Iterable of command-line arguments
    """
    # parse arguments
    argn = parse_args(args=args)
    # create HTML formatter with given style type
    formatter = HtmlFormatter(style=argn.style)

    # stdout wrapper for with statement to prevent it from being closed
    class Stdout:
        def __enter__(self):
            return sys.stdout

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    # write to file/stream + exit
    with open(argn.output_file, "w") if argn.output_file else Stdout() as f:
        print(formatter.get_style_defs(), file=f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
