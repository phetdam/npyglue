"""Replace annotated Markdown fenced code blocks with inline Pygments HTML.

.. codeauthor:: Derek Huang <djh458@stern.nyu.edu>

This program copies Markdown source line-by-line until it encounters a
specially formatted comment block placed on its own line:

    <!-- pygmentize: on -->

Then, every time a triple-backtick fenced code block is encountered, pygmentize
will be used to generate the corresponding Pygments HTML, and this is injected
into the Markdown output being copied. Therefore, we are able to control the
code block formatting style manually via HTML and use Pygments themes.

The major application is in generation of Doxygen HTML from Markdown. To stop
code block replacement, the corresponding off directive should be inserted:

    <!-- pygmentize: off -->
"""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from enum import Enum
from pathlib import Path
import re
import sys
from typing import Iterable, Optional

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import Lexer
from pygments.lexers import get_lexer_by_name

# progname name
_progname = Path(__file__).stem


class HtmlWrapping(Enum):
    """Enum to represent the HTML wrapping method.

    The following wrapping methods are as follows:

        DEFAULT     Run pygmentize as usual
        DOXYGEN     Encase pygmentize output in <pre class="fragment">
        NOWRAP      Run pygmentize with -O nowrap
    """
    DEFAULT = "default"
    DOXYGEN = "doxygen"
    NOWRAP = "nowrap"


# argument parser help
_help_input_file = "Markdown input file. If not provided, reads from stdin"
_help_output_file = "Markdown output file. If not provided, writes to stdout"
_help_wrap_html = "HTML wrapping method for the Pygments output"


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
    # input file
    ap.add_argument("-i", "--input-file", help=_help_input_file)
    # output file
    ap.add_argument("-o", "--output-file", help=_help_output_file)
    # HTML snippet wrapping method
    ap.add_argument(
        "-w",
        "--wrap-html",
        default=HtmlWrapping.NOWRAP.value,
        choices=[e.value for e in HtmlWrapping],
        help=_help_wrap_html
    )
    # parse args and return
    return ap.parse_args(args=args)


def pygmentize_block(fin, lang: str, wrap: HtmlWrapping) -> tuple[str, int]:
    """Read lines from fin until triple backticks and run pygmentize.

    This consumes input lines from the file stream, including the final triple
    backticks marking the end of the block (which is not sent to Pygments). The
    input lines are treated as a block and formatted by Pygments as HTML.

    Parameters
    ----------
        fin : file-like
            File input stream
        lang : str
            Valid Pygments language, e.g. cpp, cmake
        wrap : HtmlWrapping
            The HTML fragment wrapping style to use

    Returns
    -------
    tuple[str, int]
        Pair of the HTML fragment and the number of input lines consumed. The
        number of consumed input lines includes the triple backticks.
    """
    # list of lines to hold
    lines = []
    # read lines until triple backticks
    while True:
        line = fin.readline()
        # unexpected EOF
        if not line:
            raise RuntimeError("Unexpected end of file while handling code block")
        # triple backticks
        if line == "```\n":
            break
        # if it starts with triple backticks, error
        if line.startswith("```"):
            raise RuntimeError(f"Malformed triple backtick line:\n{line}")
        # append line
        lines.append(line)
    # join lines to form code block
    code_block = "".join(lines)
    # get lexer
    lexer = get_lexer_by_name(lang)
    # HTML formatter
    nowrap = False if wrap == HtmlWrapping.DEFAULT else True
    formatter = HtmlFormatter(nowrap=nowrap)
    # wrapping tags (only for Doxygen)
    wrap_begin = "<pre class=\"fragment\">" if wrap == HtmlWrapping.DOXYGEN else ""
    wrap_end = "</pre>" if wrap == HtmlWrapping.DOXYGEN else ""
    # return formatted HTML with delimiter tags
    # encase in <pre> with some formatting markers
    return (
        # formatted HTML
        (
            f"<!-- HTML generated with {_progname} BEGIN -->\n"
            f"{wrap_begin}{highlight(code_block, lexer, formatter)}{wrap_end}\n"
            f"<!-- HTML generated with {_progname} END -->\n"
        ),
        # number of lines consumed (including final ````)
        len(lines) + 1
    )


def main(args: Optional[Iterable[str]] = None) -> int:
    """Main function for the program.

    Parameters
    ----------
    args : Iterable[str], default=None
        Iterable of command-line arguments
    """
    # parse arguments
    argn = parse_args(args=args)

    # stdin wrapper for with statement to prevent if rom being closed
    class Stdin:
        def __enter__(self):
            return sys.stdin

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    # stdout wrapper for with statement to prevent it from being closed
    class Stdout:
        def __enter__(self):
            return sys.stdout

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    # set input + output files
    # note: in 3.10 we can use parentheses to group the statements. to support
    # < 3.10 we use lambdas to shorten the conditional open()
    def get_input():
        return open(argn.input_file) if argn.input_file else Stdin()

    def get_output():
        return open(argn.output_file, "w") if argn.output_file else Stdout()

    # HTML wrapping type
    html_wrap = HtmlWrapping(argn.wrap_html)
    # open files
    with get_input() as fin, get_output() as fout:
        # whether or not to pygmentize the next fenced code block seen
        format_block = False
        # number of lines read
        n_lines = 0
        # until completion
        while True:
            # get line. if empty, done
            line = fin.readline()
            if not line:
                break
            # read one line
            n_lines += 1
            # check for activation
            if line == "<!-- pygmentize: on -->\n":
                if format_block:
                    print(
                        f"Warning: line {n_lines}: pygmentization of fenced "
                        "code blocks already on ",
                        file=sys.stderr
                    )
                else:
                    format_block = True
                continue
            # check for deactivation
            if line == "<!-- pygmentize: off -->\n":
                if not format_block:
                    print(
                        f"Warning: line {n_lines}: pygmentization of fenced "
                        "code blocks already off ",
                        file=sys.stderr
                    )
                else:
                    format_block = False
                continue
            # if activated and matches fenced code block, process
            if format_block and re.match("```.+", line):
                # get the highlighting language + remove whitespace
                lang = line.lstrip("```").rstrip()
                # process code block with Pygments (consumes rest of block)
                fragment, n = pygmentize_block(fin, lang, html_wrap)
                # write fragment and update line count
                fout.write(fragment)
                n_lines += n
                continue
            # otherwise, write directly to fout
            fout.write(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
