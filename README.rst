.. README.rst

npyglue
=======

Repository for C++ glue code simplifying Python/C++ interop.

TBD. Requires C++17, has extra features with C++20, and simplifies manual or
SWIG_-based wrapping of C++ functions in CPython `extension modules`__ that
operate on NumPy_ arrays. Additional C++ helpers are provided to simplify use
of the `Python C API`_ and the `NumPy C array API`_ from C++ when embedding the
Python interpreter.

Also provides a facility enabling ownership of arbitrary C++ objects with
`Python capsules`_ by move construction via `placement new`_. Object type can
be queried dynamically at runtime (which requires compilation with RTTI) to
enable proper casting and dereferencing of the stored ``void*``.

.. __: https://docs.python.org/3/extending/extending.html
.. _SWIG: https://www.swig.org/
.. _NumPy: https://numpy.org/doc/stable/
.. _Python C API: https://docs.python.org/3/c-api/index.html
.. _NumPy C array API: https://numpy.org/doc/stable/reference/c-api/array.html
.. _Python capsules: https://docs.python.org/3/c-api/capsule.html
.. _placement new: https://en.cppreference.com/w/cpp/language/new#Placement_new
