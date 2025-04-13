/**
 * @file npy_capsule_test.cc
 * @author Derek Huang
 * @brief C++ program to test NumPy arrays backed by PyCapsule objects
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdlib>
#include <iostream>
#include <string_view>
#include <typeinfo>
#include <type_traits>
#include <utility>
#include <vector>

#include "npygl/features.h"
#include "npygl/ndarray.hh"
#include "npygl/python.hh"

#if NPYGL_HAS_ARMADILLO
#include <armadillo>
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_EIGEN3
#include <Eigen/Core>
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_LIBTORCH
#include <torch/torch.h>
#include "npygl/torch.hh"
#endif  // NPYGL_HAS_LIBTORCH

// TODO: consider having this not be a giant main() function

namespace {

/**
 * Traits type to indicate if a type has a `source_type` member function.
 *
 * This member function should return something that is implicitly convertible
 * to a `std::string_view`, e.g. a `const char*`, `const std::string&`.
 *
 * This traits type helps define the NumPy array object factory concept.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct has_source_type : std::false_type {};

/**
 * True specialization for a type with a `source_type` member function.
 *
 * @tparam T type
 */
template <typename T>
struct has_source_type<
  T,
  std::enable_if_t<
    std::is_convertible_v<
      decltype(std::declval<T>().source_type()),
      std::string_view
    >
  > > : std::true_type {};

/**
 * Indicate that a type satisfies the NumPy array object factory concept.
 *
 * The type must satisfy the `has_source_type<T>` traits and when invoked
 * return a `npygl::py_object` instance assumed to be a NumPy array.
 *
 * @tparam T type
 */
template <typename T>
struct is_ndarray_object_factory : std::bool_constant<
  has_source_type<T>::value && std::is_invocable_r_v<npygl::py_object, T> > {};

/**
 * Helper to indicate a type satisfies the NumPy array object factory concept.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool is_ndarray_object_factory_v = is_ndarray_object_factory<T>::value;

/**
 * SFINAE helper to indicate if a type can be used to create a NumPy array.
 *
 * @note May want to move to `ndarray.h` to improve the type traits there.
 *
 * @tparam T type
 */
template <typename T>
using make_ndarray_test_t = std::enable_if_t<
  // require move-only construction + type must have a builder
  (!std::is_reference_v<T> && npygl::can_make_ndarray_v<T>) ||
  // matches the "concept" of having a source_type() member function and when
  // invoked can return a new npygl::py_object holding a NumPy array
  is_ndarray_object_factory<T>::value
>;

/**
 * Create a NumPy array from a C++ object and print the capsule and the array.
 *
 * If there is a Python error then `npygl::py_error_exit` is called.
 *
 * @tparam T type
 *
 * @param out Stream to write output to
 * @param obj C++ object rvalue to create a NumPy array from
 */
template <typename T, typename = make_ndarray_test_t<T>>
void make_ndarray_test(std::ostream& out, T&& obj)
{
  // create NumPy array backed by C++ object
  auto ar = [&obj]
  {
    // matches concept of having source_type() member + returning py_object
    if constexpr (is_ndarray_object_factory_v<T>)
      return obj();
    // object is directly convertible
    else
      return npygl::make_ndarray(std::move(obj));
  }();
  npygl::py_error_exit();
  // get string view to type name
  std::string_view type_name = [&obj]
  {
    if constexpr (is_ndarray_object_factory_v<T>)
      return obj.source_type();
    else
      return npygl::type_name(typeid(T));
  }();
  // if NumPy object factory doesn't return a NumPy array, error
  if constexpr (is_ndarray_object_factory_v<T>) {
    if (!npygl::is_ndarray(ar)) {
      out << "-- Error: Invocation of " << npygl::type_name(typeid(T)) <<
        " instance with source_type() = " << type_name <<
        " did not produce a NumPy array" << std::endl;
      return;
    }
  }
  // get backing base object for the object
  // note: need template qualifier since ar implicitly depends on T
  auto ar_base = PyArray_BASE(ar.template as<PyArrayObject>());
  // write to stream with type name + base
  out << "-- " << type_name << " (" << ar_base << ")\n" << ar << std::endl;
}

/**
 * Create a NumPy array from a C++ object and print the capsule and the array.
 *
 * If there is a Python error then `npygl::py_error_exit` is called. This
 * overload simply writes to `std::cout` as the output stream.
 *
 * @tparam T type
 *
 * @param obj C++ object rvalue to create a NumPy array from
 */
template <typename T, typename = make_ndarray_test_t<T>>
void make_ndarray_test(T&& obj)
{
  make_ndarray_test(std::cout, std::move(obj));
}

/**
 * Functor that creates a 1D NumPy array from a tuple.
 *
 * This satisfies the `is_ndarray_object_factory<T>` traits type.
 */
struct tuple_ndarray_factory {
  constexpr auto source_type() const noexcept
  {
    // string literal is in Python type annotation style
    return "tuple[double]";
  }

  auto operator()() const noexcept
  {
    // create new tuple of doubles
    auto tup = Py_BuildValue("ddddd", 3.4, 1.222, 6.745, 5.2, 5.66, 7.333);
    npygl::py_error_exit();
    // create 1D NumPy array, taking ownership of the tuple for Py_DECREF
    auto ar = npygl::make_ndarray<double>(npygl::py_object{tup});
    npygl::py_error_exit();
    return ar;
  }
};

/**
 * Functor that creates a 1D NumPy array from a list of mixed arithmetic types.
 *
 * This satisfies the `is_ndarray_object_factory<T>` traits type.
 */
struct list_1d_ndarray_factory {
  constexpr auto source_type() const noexcept
  {
    // returned string literal is in Python type annotation style
    return "list[int | float]";
  }

  auto operator()() const noexcept
  {
    // new list of int + double
    auto list = Py_BuildValue("[dididid]", 1.22, 2, 4.11, 2, 4.132, 1, 2.22);
    npygl::py_error_exit();
    // create 1D NumPy array, taking ownership of the list for Py_DECREF
    auto ar = npygl::make_ndarray<double>(npygl::py_object{list});
    npygl::py_error_exit();
    return ar;
  }
};

/**
 * Functor that creates a 2D NumPy array from nested lists of mixed types.
 *
 * This satisfies the `is_ndarray_object_factory<T>` traits type.
 */
struct list_2d_ndarray_factory {
  constexpr auto source_type() const noexcept
  {
    // string literal is in Python type annotation style
    return "list[list[int | float]]";
  }

  auto operator()() const noexcept
  {
    // new list of lists of int + dbouel
    auto list = Py_BuildValue("[[dd][id][ii]]", 1.22, 1.32, 4, 3.22, 6, 5);
    npygl::py_error_exit();
    // create 2D NumPy array (list owned for Py_DECREF)
    auto ar = npygl::make_ndarray<double>(npygl::py_object{list});
    npygl::py_error_exit();
    return ar;
  }
};

/**
 * functor that satisfies `is_ndarray_object_factory<T>` but returns a tuple.
 */
struct tuple_factory {
  constexpr auto source_type() const noexcept
  {
    // string literal is in Python type annotation style
    return "tuple[int | double]";
  }

  auto operator()() const noexcept
  {
    return npygl::py_object{Py_BuildValue("diddi", 1.33, 9, 3.44, 2.22, 12)};
  }
};

}  // namespace

int main()
{
  // initialize Python + import NumPy API
  npygl::npy_api_import(npygl::py_init());
  npygl::py_error_exit();
  // print version
  std::cout << Py_GetVersion() << std::endl;
  // create NumPy arrays backed by double and integer vectors
  make_ndarray_test(std::vector{2.3, 1.222, 14.23, 3.243, 5.556});
  make_ndarray_test(std::vector{3, 14, 1, 555, 34, 3, 8, 42});
#if NPYGL_HAS_EIGEN3
  // create a NumPy array backed by an Eigen column-major matrix
  make_ndarray_test(
    Eigen::MatrixXf{
      {4.333f, 1.44f, 1.532f, 1.222f},
      {5.6634f, 2.2f, 1.555f, 5.64f},
      {6.7774f, 4.87f, 9.875f, 3.22f}
    }
  );
  // create a NumPy array backed by an Eigen row-major matrix
  make_ndarray_test(
    Eigen::Matrix<
      unsigned int,
      Eigen::Dynamic,
      Eigen::Dynamic,
      Eigen::StorageOptions::RowMajor
    >{
      {4, 1, 2, 3, 2},
      {6, 7, 8, 2, 3},
      {31, 6, 44, 1, 23}
    }
  );
#endif  // NPYGL_HAS_EIGEN3
#if NPYGL_HAS_ARMADILLO
  // create a NumPy array backed by an Armadillo complex matrix
  make_ndarray_test(
    arma::cx_mat{
      {{3.44, 5.423}, {9.11, 4.333}, {4.63563, 1.111}},
      {{4.23, 2.123}, {3.4244, 5.22}, {0.999, 12.213}}
    }
  );
  // create a NumPy array backed by an Armadillo complex cube
  make_ndarray_test(
    []
    {
      arma::cx_cube cube{2, 2, 3};
      // need to set slice-by-slice
      cube.slice(0) = {
        {{3.4, 2.22}, {3.22, 4.23}},
        {{5.34, 5.111}, {6.66, 1.123}}
      };
      cube.slice(1) = {
        {{6.455, 1.111}, {4.232, 0.989}},
        {{6.1212, 1.1139}, {6.45, 0.2345}}
      };
      cube.slice(2) = {
        {{1.12, 4.412}, {5.34, 6.111}},
        {{4.123, 1.998}, {8.99, 1.114}}
      };
      return cube;
    }()
  );
  // create a NumPy array backed by an Armadillo float column vector
  make_ndarray_test(arma::fvec{1.f, 3.4f, 4.23f, 3.54f, 5.223f});
  // create a NumPy array backed by an Armadillo double row vector
  make_ndarray_test(arma::rowvec{5., 4.33, 2.433, 1.22, 4.34});
#endif  // NPYGL_HAS_ARMADILLO
#if NPYGL_HAS_LIBTORCH
  // create a NumPy array backed by a random PyTorch float tensor
  make_ndarray_test(
    []
    {
      // PyTorch generator object for reproducibility. no need to acquire impl
      // mutex in single-threaded runtime like for this program
      auto gen = at::make_generator<at::CPUGeneratorImpl>();
      return torch::randn({2, 3, 4}, gen);
    }()
  );
#if NPYGL_HAS_EIGEN3
  // create a NumPy array backed by a PyTorch double tensor constructed from an
  // Eigen3 double matrix (a lot of words but very little copying)
  make_ndarray_test(
    []
    {
      Eigen::MatrixXd mat{
        {4., 1.55, 3.2, 23.1},
        {4.33, 2.145, 34.123, 7.455},
        {6.545, 1.32, 6.444, 8.75}
      };
      return npygl::make_tensor(std::move(mat));
    }()
  );
#endif  // NPYGL_HAS_EIGEN3
#endif  // NPYGL_HAS_LIBTORCH
  // create a 1D NumPy array from a tuple
  make_ndarray_test(tuple_ndarray_factory{});
  // create a 1D NumPy array from a list
  make_ndarray_test(list_1d_ndarray_factory{});
  // create a 2D NumPy array from a list of nested lists
  make_ndarray_test(list_2d_ndarray_factory{});
  // print an error because the functor doesn't create a NumPy array
  // TODO: conditionally enable it? doesn't produce an error, just a demo
  make_ndarray_test(tuple_factory{});
  return EXIT_SUCCESS;
}
