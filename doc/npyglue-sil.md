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

[SWIG]: https://www.swig.org/

Currently, the npyglue SIL consists of the following SWIG `.i` files:

<!--
    note:

    originally, the "exception handler" link reference text was `%exception`,
    but Doxygen's Markdown parser kept discarding the `%`. by the CommonMark
    standard, backslash escapes do *not* work in code blocks. using \%exception
    actually worked fine; we just could not encase it in backticks.

    so we just decided to drop the typewriter formatting and use
    "exception handler" as the link text. note that using doubld %, e.g. with
    `%%exception`, Doxygen managed to preserve one %, but this doesn't seem to
    be expected behavior per the CommonMark standard.
-->

* `ndarray.i`, which provides NumPy array SWIG
  [typemap](https://www.swig.org/Doc4.0/Python.html#Python_nn53) macros
* `python.i`, which provides a SWIG C++
  [exception handler](https://www.swig.org/Doc4.0/Python.html#Python_nn44)

Please read the Doxygen-style comment blocks in each `.i` file for detailed
documentation.

> **Note**
>
> npyglue's NumPy interop functionality in SWIG interfaces is still rather
> limited. The `ndarray_flat_view<T>` template only provides a flat, unstrided
> view of a NumPy array, and is not sufficient for functions that expected a
> certain NumPy array dimensionality as input. This is a later planned feature.

> **Warning**
>
> Do *not* use the `ndarray_flat_view<T>` in/out typemaps as their semantics
> are somewhat confusing. The current semantics were likely from an early stage
> in the project's development where `make_ndarray(T&&)` function template did
> not yet exist to allow data buffer ownership to be transferred from C++ to a
> NumPy array without any data copy.

## Walkthrough

Let's walk through an example. Suppose we have a C++ function \[template\]
called `xmath::normal` that will return a `std::vector<T>` of standard normal
variates. It is defined in `xmath/random.hh`, whose contents are as follows:

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

[NumPy]: https://numpy.org/doc/stable

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

This `.i` file essentially does the following few things:

1. Enables the provided C++ exception handler
2. Enables ["out" typemaps](https://www.swig.org/Doc4.0/Typemaps.html#Typemaps_nn28)
   that convert appropriate C++ objects into NumPy arrays
3. Individually wrap function template instantiations using the SWIG C++
   [template directive](https://www.swig.org/Doc4.0/SWIGPlus.html#SWIGPlus_template_directive)

Now let's prepare to build our SWIG-generated Python extension module. There
are a few ways to do this, but we will use [CMake], as from experience it is
easier to integrate Python extension module building into an existing CMake C++
build than to attempt the reverse via [setuptools], for example.

[CMake]: https://cmake.org/cmake/help/latest/
[setuptools]: https://setuptools.pypa.io/en/latest/

For simplicity we make the following assumptions:

1. `xmath` is a separately installed library with `find_package` support,
   exporting an `INTERFACE` target `xmath::xmath`
2. We are using a
   [single-config](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#build-configurations)
   CMake generator, e.g. Make or Ninja

To this end, let's write the following bare-bones `CMakeLists.txt`:

<!--
    note:

    Doxygen only supports Markdown code block highlighting for the languages it
    knows how to process. we therefore have some choices which are ranked from
    hackiest/easiest to most work/best Doxygen integration.

    1. highlight CMake as Python so comments are at least a different color.
       this means that browsing the Markdown files will look weird, however.
    2. use Pygments' pygmentize to generate a CSS stylesheet and inject HTML
       to load the stylesheet and HTML fragment into a Markdown file. this is a
       infile-pygmentized.md file generated from a infile.md file.

       to generate the CSS stylesheet we use:

          pygmentize -S one-dark -f html -O classprefix=pygments-

       to convert a CMake snippet from a file we use:

          pygmentize -l cmake -f html -O nowrap -o outfile infile

       we use nowrap option to not use a <div> and <pre> and instead use the
       <pre class="fragment">, whose formatting is provided by Doxygen, to wrap
       the HTML that we intend to embed into the Markdown file.

       this produces Pygments highlighting whose style we can control with the
       CSS stylesheet (differen -S option) that essentially looks like a
       Doxygen code block but without links and with different highlighting.

       we curently can accomplish this with the tools/pyginject.py script. it
       must be run with the -w doxygen option for correct code block formatting
       but still has some issues with escaping % for example. the generated CSS
       stylesheet can then be added to HTML_EXTRA_STYLESHEET. there will be a
       CMake add_custom_command to invoke this transform step and hang it as
       part of the overall build system. there can be a finalization target
       that depends on all the outputs that the Doxygen target depends on.

    3. write our own Doxygen filter to translate CMake into pseudo-C. this
       seems like a decent amount of work and it is not clear if Doxygen will
       even be able to apply the filter to fenced code blocks.
-->

<!-- pygmentize: on -->

```cmake
cmake_minimum_required(VERSION 3.20)

project(xmath-python VERSION 0.1.0 LANGUAGES CXX)

# find Python 3 development artifacts + ensure the environment has NumPy
find_package(Python3 3.8 REQUIRED COMPONENTS Development NumPy)
# locate SWIG >= 4.0 with Python wrapping support + enable
find_package(SWIG 4.0 REQUIRED COMPONENTS python)
include(UseSWIG)
# find xmath
find_package(xmath REQUIRED)
# find npyglue
# TODO: not sure if SIL should be considered a separate component
find_package(npyglue 0.1.0 REQUIRED)
# add Python C++ extension module generated via SWIG
set_property(SOURCE xmath_random.i PROPERTY CPLUSPLUS ON)
swig_add_library(
    xmath_random
    LANGUAGE python
    # write xmath_random.py and the generated C++ source in the build directory
    OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OUTFILE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCES xmath_random.i
)
# ensure SWIG picks up npyglue .i include path
npygl_swig_include(xmath_random TARGETS npyglue::SIL xmath::xmath)
```

<!-- pygmentize: off -->

TBD
