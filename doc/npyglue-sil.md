# npyglue SWIG Interface Library

<!--
    npyglue-sil.md

    Author: Derek Huang
    License: MIT License

    This Markdown file is written in GitHub Flavored Markdown.

    Note:

    The [!NOTE] alert blocks are only supported by Doxygen 1.11.0 and above so
    we do not use them in the documentation (may be changed later).

    Markdown has no official support for comments and we want Doxygen to use
    the level 1 heading as the page title, so this HTML comment block is placed
    below, instead of above, the level 1 heading serving as the title.
-->

npyglue comes with [SWIG] interface files to simplify the creation of Python
extension modules with SWIG. These SWIG interface files work in tandem with the
C++ headers and are collectively referred to as the npyglue SWIG Interface
Library (SIL).

Currently, the npyglue SIL consists of the following SWIG `.i` files:

* `ndarray.i`, which provides NumPy array SWIG
  [typemap](https://www.swig.org/Doc4.0/Python.html#Python_nn53) macros
* `python.i`, which provides a SWIG C++
  [`%exception`](https://www.swig.org/Doc4.0/Python.html#Python_nn44) handler

Please read the Doxygen-style comment blocks in each `.i` file for detailed
documentation.

> Note:
>
> npyglue's NumPy interop functionality in SWIG interfaces is still rather
> limited. The `ndarray_flat_view<T>` template only provides a flat, unstrided
> view of a NumPy array, and is not sufficient for functions that expected a
> certain NumPy array dimensionality as input. This is a later planned feature.

> Warning:
>
> Do *not* use the `ndarray_flat_view<T>` in/out typemaps as their semantics
> are somewhat confusing. The current semantics were likely from an early stage
> in the project's development where `make_ndarray(T&&)` function template did
> not yet exist to allow data buffer ownership to be transferred from C++ to a
> NumPy array without any data copy.

## Walkthrough

Let's walk through an example. Suppose we have a C++ function [template] called
`xmath::normal` that will return a `std::vector<T>` of standard normal variates.
It is defined in `xmath/random.hh`, whose contents are as follows:

<!--
    note:

    technically c++ can be used as the language in the fenced code block but
    Doxygen seems to only parse the "c" part and "++" will be part of the code.
-->

```cpp
// xmath/random.hh

#include <cstdint>
#include <optional>
#include <random>
#include <type_traits>

namespace xmath {

// helper traits for valid floating point types. in C++23 is_floating_point<T>
// also accepts the extended floating point types like the bfloat16_t
template <typename T, typename = void>
struct is_std_float : std::false_type {};

template <typename T>
struct is_std_float<
  T,
  std::enable_if_t<
    std::is_same_v<T, float> ||
    std::is_same_v<T, double> ||
    std::is_same_v<T, long double>
  > > : std::true_type {};

// SFINAE helper
template <typename T>
using std_float_t = std::enable_if_t<is_std_float<T>::value>;

// seed type optional
using optional_seed_type = std::optional<std::uint_fast32_t>;

// return vector of normal variates (optionally seeded)
template <typename T, typename = std_float_t<T>>
auto normal(std::size_t n, optional_seed_type seed = {})
{
    // PRNG generator
    std::mt19937 rng{seed ? *seed : std::random_device{}()};
    // construct distribution and fill vector
    std::normal_distribution dist;
    std::vector<T> values(n);
    for (decltype(n) i = 0; i < n; i++)
        values[i] = dist(rng);
    return values;
}

}  // namespace xmath
```

If we want to expose `xmath::normal` from Python then we need to build
a [Python extension module](https://docs.python.org/3/extending/extending.html).
Furthermore, we want interop with [NumPy], as it is commonly used in the Python
CPU-based scientific stack. We can do this by manually writing a Python
extension module, which is greatly simplified with npyglue, but some projects
prefer to use SWIG for wholesale target language wrapper generation. This is
where the npyglue SIL comes into play.

Let's write a simple SWIG interface file called `xmath_random.i`:

```cpp
// xmath_random.i

%{
#define SWIG_FILE_WITH_INIT  // tell SWIG this is a standalone module

#include <cstdint>
#include <vector>

#include <npygl/ndarray.hh>  // includes <numpy/ndarrayobject.h>

#include "xmath/random.hh"
%}

%include "npygl/ndarray.i"  // for NPYGL_APPLY_NDARRAY_OUT_TYPEMAP
%include "npygl/python.i"   // for npyglue SWIG %exception handler

// enable the npyglue-provided C++ exception handler
NPYGL_ENABLE_EXCEPTION_HANDLER

// apply std::vector<T> NumPy array typemaps
NPYGL_APPLY_NDARRAY_OUT_TYPEMAP(std::vector<double>);
NPYGL_APPLY_NDARRAY_OUT_TYPEMAP(std::vector<float>);
NPYGL_APPLY_NDARRAY_OUT_TYPEMAP(std::vector<long double>);

%inline %{
namespace xmath {

// wrapper for normal for Python
// note: when we have std::optional<T> typemaps working this is unnecessary
// note: SWIG doesn't understand auto so we need to write out the return type
template <typename T>
std::vector<T> normal(std::size_t n, int seed = -1)
{
  return normal(
    n,
    (seed < 0) ? optional_seed_type{} : optional_seed_type{seed}
  );
}

}  // namespace xmath
%}

namespace xmath {

// each template must be instantiated
%template(normal) normal<double>;
%template(fnormal) normal<float>;
%template(ldnormal) normal<long double>;

}  // namespace xmath
```

TBD, add content for SWIG/CMake integration.

[SWIG]: https://www.swig.org/
[NumPy]: https://numpy.org/doc/stable
