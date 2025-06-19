# npyglue

<!--
    README.md

    Author: Derek Huang
    License: MIT License

    This Markdown file is written in GitHub Flavored Markdown.

    Note:

    Originally this file was written in reStructuredText but for compatibility
    with Doxygen was unfortunately rewritten in Markdown.

    Markdown has no official support for comments and we want Doxygen to use
    the level 1 heading as the page title, so this HTML comment block is placed
    below, instead of above, the level 1 heading serving as the title.
-->

A header-only C++ library simplifying Python/C++ interop.

TBD. Requires C++17, has extra features with C++20, and simplifies manual or
[SWIG]-based wrapping of C++ functions in CPython [extension modules] that
operate on [NumPy] arrays. Additional C++ helpers are provided to simplify use
of the [Python C API] and the [NumPy Array C API] from C++ when embedding the
Python interpreter.

Also provides a facility enabling ownership of arbitrary C++ objects with
[Python capsules] by move construction via [placement `new`][placement new].
Object type can be queried dynamically at runtime (which requires compilation
with RTTI) so the stored `void*` can be cast and dereferenced safely inside
functions for usage in C++ layer.

[extension modules]: https://docs.python.org/3/extending/extending.html
[SWIG]: https://www.swig.org/
[NumPy]: https://numpy.org/doc/stable/
[Python C API]: https://docs.python.org/3/c-api/index.html
[NumPy Array C API]: https://numpy.org/doc/stable/reference/c-api/array.html
[Python capsules]: https://docs.python.org/3/c-api/capsule.html
[placement new]: https://en.cppreference.com/w/cpp/language/new#Placement_new
