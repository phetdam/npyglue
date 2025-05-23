/**
 * @file python/argparse.hh
 * @author Derek Huang
 * @brief C++ header for Python argument parsing helpers
 * @copyright MIT License
 */

#ifndef NPYGL_PYTHON_ARGPARSE_HH_
#define NPYGL_PYTHON_ARGPARSE_HH_

#ifndef PY_SSIZE_T_CLEAN
#define PY_SSIZE_T_CLEAN
#endif  // PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "npygl/common.h"

namespace npygl {

/**
 * Traits type for providing the Python argument parsing format for a type.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct py_format_type;

/**
 * Placeholder type to indicate that subsequence format units are optional.
 */
struct py_optional_args {};

/**
 * Define a Python format type specialization for a single type.
 *
 * This provides a null-terminated char array format string and a length value.
 *
 * @param type C type
 * @param fmt Format string specification for `PyArg_ParseTuple`
 */
#define NPYGL_PY_FORMAT_TYPE_SPEC(type, fmt) \
  template <> \
  struct py_format_type<type> { \
    static constexpr const char value[] = fmt; \
    static constexpr std::size_t length = sizeof value - 1; \
  }

// see https://docs.python.org/3/c-api/arg.html for formatting details
NPYGL_PY_FORMAT_TYPE_SPEC(const char*, "s");
NPYGL_PY_FORMAT_TYPE_SPEC(unsigned char, "b");
NPYGL_PY_FORMAT_TYPE_SPEC(short, "h");
NPYGL_PY_FORMAT_TYPE_SPEC(int, "i");
// note: no long or long long because Py_ssize_t can be either
// NPYGL_PY_FORMAT_TYPE_SPEC(long, "l");
// NPYGL_PY_FORMAT_TYPE_SPEC(long long, "L");
NPYGL_PY_FORMAT_TYPE_SPEC(Py_ssize_t, "n");
NPYGL_PY_FORMAT_TYPE_SPEC(float, "f");
NPYGL_PY_FORMAT_TYPE_SPEC(double, "d");
NPYGL_PY_FORMAT_TYPE_SPEC(Py_complex, "D");
NPYGL_PY_FORMAT_TYPE_SPEC(PyObject*, "O");
NPYGL_PY_FORMAT_TYPE_SPEC(PyBytesObject*, "S");
NPYGL_PY_FORMAT_TYPE_SPEC(py_optional_args, "|");
NPYGL_PY_FORMAT_TYPE_SPEC(Py_buffer, "y*");

/**
 * Partial specialization for a single type to terminate template instantation.
 *
 * @tparam Is... Indices from 0 through `py_format_type<T>::length - 1`
 * @tparam Js... Indices from 0 (unused)
 * @tparam T Type to get format string for
 */
template <std::size_t... Is, std::size_t... Js, typename T>
struct py_format_type<std::index_sequence<Is...>, std::index_sequence<Js...>, T>
  : py_format_type<T> {};

/**
 * Partial specialization for building the format string for multiple types.
 *
 * The `value` character array is built recursively via index pack expansion.
 *
 * @tparam Is... Indices from 0 through `py_format_type<T>::length - 1`
 * @tparam Js... Indices from 0 through `py_format_type<Ts...>::length - 1`
 * @tparam T Type to get format string for
 * @tparam Ts... Remaining types to get format string for
 */
template <std::size_t... Is, std::size_t... Js, typename T, typename... Ts>
struct py_format_type<
  std::index_sequence<Is...>,
  std::index_sequence<Js...>,
  T,
  Ts...
> {
  static constexpr const char value[] = {
    py_format_type<T>::value[Is]...,
    py_format_type<Ts...>::value[Js]...,
    '\0'
  };
  static constexpr auto length = sizeof value - 1;
};

/**
 * Partial specialization for providing the format string for multiple types.
 *
 * This uses the index sequence partial specializations to provide members.
 *
 * @tparam T First type
 * @tparam Ts... Remaining types
 */
template <typename T, typename... Ts>
struct py_format_type<T, Ts...> : py_format_type<
  std::make_index_sequence<py_format_type<T>::length>,
  std::make_index_sequence<py_format_type<Ts...>::length>,
  T,
  Ts...
> {};

/**
 * Partial specialization when specifying the types using a tuple.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
struct py_format_type<std::tuple<Ts...>> : py_format_type<Ts...> {};

/**
 * Compile-time Python argument format string.
 *
 * This provides a reference to a null-terminated character array.
 *
 * @tparam Ts... types
 */
template <typename... Ts>
inline constexpr const auto& py_format = py_format_type<Ts...>::value;

/**
 * Parse Python arguments into the given variadic arguments.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam Ts... types
 *
 * @param args Python arguments
 * @param vars Variables to parse arguments into
 * @returns `true` on success, `false` on error
 */
template <typename... Ts>
bool parse_args(PyObject* args, Ts&... vars) noexcept
{
  return !!PyArg_ParseTuple(args, py_format<Ts...>, &vars...);
}

namespace detail {

/**
 * Parse Python arguments into the given variable references.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam Ts... types
 * @tparam Is... Index values from 0 through sizeof...(Ts) - 1
 *
 * @param args Python arguments
 * @param vars Variable references to parse arguments into
 * @param var_is Unused index sequence to deduce indices
 * @returns `true` on success, `false` on error
 */
template <typename... Ts, std::size_t... Is>
bool parse_args(
  PyObject* args,
  const std::tuple<Ts&...>& vars,
  std::index_sequence<Is...> /*var_is*/) noexcept
{
  static_assert(sizeof...(Ts) == sizeof...(Is));
  return !!PyArg_ParseTuple(args, py_format<Ts...>, &std::get<Is>(vars)...);
}

}  // namespace detail

/**
 * Parse Python arguments into the given variable references.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam Ts... types
 *
 * @param args Python arguments
 * @param vars Variable references to parse arguments into
 * @returns `true` on success, `false` on error
 */
template <typename... Ts>
bool parse_args(PyObject* args, const std::tuple<Ts&...>& vars) noexcept
{
  return detail::parse_args(args, vars, std::index_sequence_for<Ts...>{});
}

namespace detail {

/**
 * Parse Python arguments into the given variable references.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam RTs... Required types
 * @tparam RIs... Index values from 0 through sizeof...(RTs) - 1
 * @tparam OTs... Optional types
 * @tparam OIs... Index values from 0 through sizeof...(OTs) - 1
 *
 * @param args Python arguments
 * @param reqs Variable references to parse required arguments into
 * @param req_is Unused index sequence to deduce indices
 * @param opts Varuable references to parse optional arguments into
 * @param opt_is Unused index sequence to deduce indices
 * @returns `true` on success, `false` on error
 */
template <typename... RTs, std::size_t... RIs, typename... OTs, std::size_t... OIs>
bool parse_args(
  PyObject* args,
  const std::tuple<RTs&...>& reqs,
  std::index_sequence<RIs...> /*req_is*/,
  const std::tuple<OTs&...>& opts,
  std::index_sequence<OIs...> /*opt_is*/) noexcept
{
  static_assert(sizeof...(RTs) == sizeof...(RIs));
  static_assert(sizeof...(OTs) == sizeof...(OIs));
  return !!PyArg_ParseTuple(
    args,
    py_format<RTs..., py_optional_args, OTs...>,
    &std::get<RIs>(reqs)...,
    &std::get<OIs>(opts)...
  );
}

}  // namespace detail

/**
 * Parse Python arguments into the given variable references.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam RTs... Required types
 * @tparam OTs... Optional types
 *
 * @param args Python arguments
 * @param reqs Variable references to parse required arguments into
 * @param opts Varuable references to parse optional arguments into
 * @returns `true` on success, `false` on error
 */
template <typename... RTs, typename... OTs>
bool parse_args(
  PyObject* args,
  const std::tuple<RTs&...>& reqs,
  const std::tuple<OTs&...>& opts) noexcept
{
  return detail::parse_args(
    args,
    reqs,
    std::index_sequence_for<RTs...>{},
    opts,
    std::index_sequence_for<OTs...>{}
  );
}

namespace detail {

/**
 * Helper that expands a pack of indices into the empty string.
 *
 * @tparam Is... Indexing pack
 */
template <std::size_t... Is>
inline constexpr const char* empty_string = "";

/**
 * Parse Python arguments into the given variable references.
 *
 * @note This function is only for `METH_VARARGS | METH_KEYWORDS` functions.
 *
 * @tparam RTs... Required types
 * @tparam RIs... Index values from 0 through sizeof...(RTs) - 1
 * @tparam N Number of keyword arguments
 * @tparam OTs... Optional types
 * @tparam OIs... Index values from 0 through sizeof...(OTs) - 1
 *
 * @param args Python required arguments
 * @param reqs Variable references to parse required arguments into
 * @param req_is Unused index sequence to deduce indices
 * @param kws Optional keywords argument names
 * @param kwargs Python keyword optional arguments
 * @param opts Variable references to parse optional keywords arguments into
 * @param req_is Unused index sequence to deduce indices
 * @returns `true` on success, `false` on error
 */
template <
  typename... RTs,
  std::size_t... RIs,
  std::size_t N,
  typename... OTs,
  std::size_t... OIs>
bool parse_args(
  PyObject* args,
  const std::tuple<RTs&...>& reqs,
  std::index_sequence<RIs...> /*req_is*/,
  const char* (&kws)[N],
  PyObject* kwargs,
  const std::tuple<OTs&...>& opts,
  std::index_sequence<OIs...> /*opt_is*/) noexcept
{
  // counts of required and optional types
  constexpr auto n_req = sizeof...(RTs);
  constexpr auto n_opt = sizeof...(OTs);
  // sanity checks
  static_assert(n_req == sizeof...(RIs));
  static_assert(n_opt == sizeof...(OIs));
  static_assert(n_opt == N);
  // construct array of names. the positional args we force to be positional
  // only by using "" and zero everything out. +1 for terminating nullptr
  const char* names[n_req + n_opt + 1] = {empty_string<RIs>...};
  // use fold expression to iterate over pack and set kwarg names
  ((names[n_req + OIs] = kws[OIs]), ...);
  // parse args and kwargs
  return !!PyArg_ParseTupleAndKeywords(
    args,
    kwargs,
    py_format<RTs..., py_optional_args, OTs...>,
    (char**) names,
    &std::get<RIs>(reqs)...,
    &std::get<OIs>(opts)...
  );
}

}  // namespace detail

/**
 * Parse Python arguments into the given variable references.
 *
 * @note This function is only for `METH_VARARGS | METH_KEYWORDS` functions.
 *
 * @tparam RTs... Required types
 * @tparam N Number of keyword arguments
 * @tparam OTs... Optional types
 *
 * @param args Python required arguments
 * @param reqs Variable references to parse required arguments into
 * @param kws Optional keywords argument names
 * @param kwargs Python keyword optional arguments
 * @param opts Varuable references to parse optional keywords arguments into
 * @returns `true` on success, `false` on error
 */
template <typename... RTs, std::size_t N, typename... OTs>
bool parse_args(
  PyObject* args,
  const std::tuple<RTs&...>& reqs,
  const char* (&kws)[N],
  PyObject* kwargs,
  const std::tuple<OTs&...>& opts) noexcept
{
  return detail::parse_args(
    args,
    reqs,
    std::index_sequence_for<RTs...>{},
    kws,
    kwargs,
    opts,
    std::index_sequence_for<OTs...>{}
  );
}

namespace detail {

/**
 * Traits type for providing a compile-time `PyObject*[N]` format string.
 *
 * We need to provide an empty base since a partial specialization is used.
 *
 * @tparam T type
 */
template <typename T>
struct py_object_format_type {};

/**
 * Partial specialization for getting an index sequence's parameter pack.
 *
 * This uses the `O` format character to build the format string.
 *
 * We expand the parameter pack to construct a null-terminated character array
 * that can be used as a `PyArg_Parse*` or `Py_BuildValue` format string.
 *
 * @tparam Is... Indices from the `std::index_sequence<Is...>`
 */
template <std::size_t... Is>
struct py_object_format_type<std::index_sequence<Is...>> {
  // Is - Is is used to involve the index parameter pack
  static constexpr const char value[] = {('O' + (Is - Is))..., '\0'};
};

}  // namespace detail

/**
 * Traits type for providing a compile-time `PyObject*[N]` format string.
 *
 * @tparam N Number of `PyObject*` to format with `O`
 */
template <std::size_t N>
struct py_object_format_type
  : detail::py_object_format_type<std::make_index_sequence<N>> {
  static_assert(N, "N must be nonzero for a valid format string");
};

/**
 * Compile-time `PyObject*[N]` format string.
 *
 * @tparam N Number of `PyObject*` to format with `O`
 */
template <std::size_t N>
inline constexpr const char* py_object_format = py_object_format_type<N>::value;

/**
 * Parse Python arguments into an array of `PyObject` pointers.
 *
 * This uses the generic `"O"` conversion specifier with `PyArg_ParseTuple`.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam N Number of expected Python arguments
 * @tparam Is... Sequence of unique array indices within 0 to N - 1 inclusive
 *
 * @param args Python arguments
 * @param objs Array of `PyObject*` to convert to
 * @param seq Index sequence indicating which elements of `objs` are populated
 * @returns `true` on success, `false` on error
 */
template <std::size_t N, std::size_t... Is>
bool parse_args(
  PyObject* args,
  PyObject* (&objs)[N],
  std::index_sequence<Is...> NPYGL_UNUSED(seq)) noexcept
{
  // number of indices must be less than or equal to array size
  static_assert(sizeof...(Is), "at least one index must be provided");
  static_assert(sizeof...(Is) <= N, "index count cannot exceed array size");
  // ensure none of the indices are outside of the array
  // note: parentheses around Is < N are unnecessary but are just for clarity
  static_assert(
    std::conjunction_v<std::bool_constant<(Is < N)>...>,
    "indices must only index within the provided array"
  );
  // parse args. we need to use sizeof...(Is) since it may not equal N
  return !!PyArg_ParseTuple(args, py_object_format<sizeof...(Is)>, &objs[Is]...);
}

/**
 * Parse Python arguments into an array of `PyObject` pointers.
 *
 * This uses the generic `"O"` conversion specifier with `PyArg_ParseTuple`.
 *
 * @note This function is intended for use with `METH_VARARGS` functions only.
 *
 * @tparam N Number of expected Python arguments
 *
 * @param args Python arguments
 * @param objs Array of `PyObject*` to convert to
 * @returns `true` on success, `false` on error
 */
template <std::size_t N>
inline bool parse_args(PyObject* args, PyObject* (&objs)[N]) noexcept
{
  return parse_args(args, objs, std::make_index_sequence<N>{});
}

}  // namespace npygl

#endif  // NPYGL_PYTHON_ARGPARSE_HH_
