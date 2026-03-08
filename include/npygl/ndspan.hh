/**
 * @file ndspan.hh
 * @author Derek Huang
 * @brief C++ header for n-dimensional view type
 * @copyright MIT License
 */

#ifndef NPYGL_NDSPAN_HH_
#define NPYGL_NDSPAN_HH_

#include <cstddef>
#include <ostream>
#include <type_traits>
#include <utility>

namespace npygl {

/**
 * Traits type for an extent type that provides index computation.
 *
 * @tparam T type
 */
template <typename T, typename = void>
struct is_extent_type : std::true_type {};

/**
 * Tag type to indicate selection of a range.
 *
 * This could be used to represent selecting an axis or part of a range.
 */
struct range_select_type {};

/**
 * Tag global to indicate selection of a range or axis.
 */
inline constexpr range_select_type all_;

namespace detail {

/**
 * Traits helper to enforce ctor constraints on a pack of extents/indices.
 *
 * Each type must be an integral type and `sizeof...(Ts)` must be <= `N` - 1.
 *
 * @note See `ndeoo` for details on the naming.
 *
 * @tparam N Number of dimensions
 * @tparam T First type
 * @tparam Ts Subsequent types
 */
template <std::size_t N, typename T, typename... Ts>
constexpr bool ndeoo_ctor_constraint = (
  // implicitly also ensures that N >= 1 itself
  (sizeof...(Ts) + 1u <= N) &&
  // check T and Ts... pack types are all integral
  std::is_integral_v<T> && (std::is_integral_v<Ts> && ...)
);

}  // namespace detail

// forward decl for adjust_extent_t
template <std::size_t N>
class ndextents;

// forward decl for adjust_extent_t
template <std::size_t N>
class ndoffsets;

namespace detail {

/**
 * Produce a type with the given extent.
 *
 * Each `E` specialization must produce a type member `type`.
 *
 * @tparam E Extent type
 * @tparam N Number of dimensions
 */
template <typename E, std::size_t N>
struct adjust_extent {};

/**
 * Partial specialization for `ndextent<N>`.
 *
 * @tparam N_ Number of original dimensions
 * @tparam N Number of target dimensions
 */
template <std::size_t N_, std::size_t N>
struct adjust_extent<ndextents<N_>, N> {
  using type = ndextents<N>;
};

/**
 * Partial specialization for `ndoffsets<N>`
 *
 * @tparam N_ Number of original dimensions
 * @tparam N Number of target dimensions
 */
template <std::size_t N_, std::size_t N>
struct adjust_extent<ndoffsets<N_>, N> {
  using type = ndoffsets<N>;
};

/**
 * SFINAE helper to obtain an extent type with adjusted dimension.
 *
 * @tparam E Extent type
 * @tparam N Number of dimensions
 */
template <typename E, std::size_t N>
using adjust_extent_t = typename adjust_extent<E, N>::type;

}  // namespace detail

/**
 * N-dimensional view extents/offsets CRTP base type.
 *
 * This represents the dimensions of a N-dimensional view and is similar to a
 * `std::array<T, N>` but provides some other features like slicing, adding and
 * subtracting extents or offsets from each other, etc.
 *
 * @note `ndeoo` is an acronym for "n-dimensional extents or offsets"
 *
 * @tparam E Extent or offsets type
 * @tparam N Number of dimensions
 * @tparam D Ctor defaulting value
 */
template <typename E, std::size_t N, std::size_t D>
class ndeoo {
public:
  // note: we don't want D to be anything other than 0 or 1
  static_assert(D == 0u || D == 1u, "D must be 0 or 1");

  /**
   * Ctor.
   *
   * Constructs the dimensions from individual arguments. If `sizeof...(Ts)` is
   * less than `N`, the remaining elements are set to `D`.
   *
   * @tparam Ts Integral types
   *
   * @param vals Extents or offsets
   */
  template <
    typename... Ts,
    typename = std::enable_if_t<detail::ndeoo_ctor_constraint<N, Ts...>> >
  constexpr ndeoo(Ts... vals) noexcept
    : ndeoo{std::index_sequence_for<Ts...>{}, vals...}
  {}

  /**
   * Ctor.
   *
   * Constructs the dimensions from an array of extents or indices.
   *
   * @param vals Extents or offsets
   */
  constexpr ndeoo(const std::size_t (&vals)[N]) noexcept
  {
    for (decltype(N) i = 0u; i < N; i++)
      vals_[i] = vals[i];
  }

  /**
   * Return the `i`th element of the extents or offsets.
   *
   * @param i Dimension index
   */
  constexpr auto& operator[](std::size_t i) noexcept
  {
    return vals_[i];
  }

  /**
   * Return the `i`th element of the extents or offsets.
   *
   * @param i Dimension index
   */
  constexpr auto& operator[](std::size_t i) const noexcept
  {
    return vals_[i];
  }

  /**
   * Return an iterator to the first element in the extents or offsets.
   */
  constexpr auto begin() noexcept
  {
    return vals_;
  }

  /**
   * Return an iterator to the first element in the extents or offsets.
   */
  constexpr auto begin() const noexcept
  {
    return vals_;
  }

  /**
   * Return an iterator to one past the last extent or offset.
   */
  constexpr auto end() noexcept
  {
    return vals_ + N;
  }

  /**
   * Return an iterator to one past the last extent or offset.
   */
  constexpr auto end() const noexcept
  {
    return vals_ + N;
  }

  /**
   * Compare against another set of extents or indices for equality.
   *
   * @param other Other extents or indices
   */
  constexpr bool operator==(const ndeoo& other) const noexcept
  {
    for (decltype(N) i = 0u; i < N; i++)
      if (vals_[i] != other.vals_[i])
        return false;
    return true;
  }

  /**
   * Add two sets of extents or indices together.
   *
   * @param other Other extents or indices to add
   */
  constexpr auto operator+(const ndeoo& other) const noexcept
  {
    // copy of extents/offsets
    auto res = *this;
    // update + return
    for (decltype(N) i = 0u; i < N; i++)
      res[i] += other.vals_[i];
    return res;
  }

  /**
   * Subtract extents or indices from the current extents or indices.
   *
   * @param other Other extents or indices to subtract
   */
  constexpr auto operator-(const ndeoo& other) const noexcept
  {
    // copy of extents/offsets
    auto res = *this;
    // update + return
    for (decltype(N) i = 0u; i < N; i++)
      res[i] -= other.vals_[i];
    return res;
  }

  /**
   * Return a slice of the extent or offset values.
   *
   * For example, let `dims` be a `dimensions<5u>`. Then, the expression
   * `dims.slice<1u>()` could be used to get a `dimensions<4u>` starting from
   * the second dimension, and `dims.slice<2u, 4u>` could be used to get a
   * `dimensions<2u>` starting from the third dimension.
   *
   * @tparam I Index of starting dimension
   * @tparam J Index one past the ending dimension
   */
  template <std::size_t I, std::size_t J = N>
  constexpr auto slice() const noexcept
  {
    static_assert(I < N, "I must be less than N");
    static_assert(J <= N, "J must be less than or equal to N");
    static_assert(I < J, "J must be greater than I");
    return slice<I, J>(std::make_index_sequence<J - I>{});
  }

  /**
   * Return the extents or offsets in reversed order.
   */
  constexpr auto reverse() const noexcept
  {
    std::size_t vals[N];
    for (decltype(N) i = 0u; i < N; i++)
      vals[i] = vals_[N - i - 1u];
    return ndeoo{vals};
  }

private:
  std::size_t vals_[N];

  /**
   * Ctor.
   *
   * Implements construction from individual extent or index values. For any
   * unspecified dimensions they are simply set to `D`.
   *
   * @tparam Is Indices 0 through `sizeof...(Ts)` - 1
   * @tparam Ts `std::size_t`
   *
   * @param vals Extents of offsets
   */
  template <std::size_t... Is, typename... Ts>
  constexpr ndeoo(std::index_sequence<Is...>, Ts... vals)
  {
    static_assert(sizeof...(Is) == sizeof...(Ts), "index and dim mismatch");
    // assign specified dimensions
    ((vals_[Is] = vals), ...);
    // for unspecified dimensions fill with default
    for (std::size_t i = sizeof...(Is); i < N; i++)
      vals_[i] = D;
  }

  /**
   * Return a slice of the extent or offset values.
   *
   * @tparam I Index of starting dimension
   * @tparam J Index one past the ending dimension
   * @tparam Is Indices 0 through J - I - 1
   */
  template <std::size_t I, std::size_t J, std::size_t... Is>
  constexpr auto slice(std::index_sequence<Is...>) const noexcept
  {
    // total number of dimensions in slice
    constexpr auto N_ = J - I;
    static_assert(sizeof...(Is) == N_, "too many dimensions in slice");
    // return slice
    // note: requires adjust_extent<E, N> specialization
    return detail::adjust_extent_t<E, N_>{vals_[I + Is]...};
  }
};

/**
 * N-dimensional view dimensions type.
 *
 * Any omitted dimension extents are defaulted to 1.
 *
 * @tparam N Number of dimensions
 */
template <std::size_t N>
class ndextents : public ndeoo<ndextents<N>, N, 1u> {
public:
  using base_type = ndeoo<ndextents<N>, N, 1u>;
  using base_type::base_type;
};

// user-defined deduction guide
template <typename... Ts>
ndextents(Ts...) -> ndextents<sizeof...(Ts)>;

/**
 * N-dimensional view index type.
 *
 * Any omitted index offsets are defaulted to 0.
 *
 * @tparam N Number of dimension
 */
template <std::size_t N>
class ndoffsets : public ndeoo<ndoffsets<N>, N, 0u> {
public:
  using base_type = ndeoo<ndoffsets<N>, N, 0u>;
  using base_type::base_type;
};

// user-defined deduction guide
template <typename... Ts>
ndoffsets(Ts...) -> ndoffsets<sizeof...(Ts)>;

/**
 * Traits type indicating the number of dimensions in an extent type.
 *
 * Each valid extent type satisfying `is_extent_type<T>` must implement this by
 * providing a `static constexpr std::size_t` member named `value`.
 *
 * @tparam T type
 */
template <typename T>
struct extent_of {};

/**
 * Number of dimensions in an extent type.
 *
 * @tparam T type
 */
template <typename T>
constexpr std::size_t extent_of_v = extent_of<T>::value;

/**
 * True specialization for an extent type providing index computation.
 *
 * A valid extent type must be trivially initializable and be invokable on a
 * dimension and stride array of the same length.
 *
 * @tparam T Extent type
 */
template <typename T>
struct is_extent_type<
  T,
  std::enable_if_t<
    // T{} must be trivial
    std::is_trivially_default_constructible_v<T> &&
    // T{}(/* array */, /* array */)
    std::is_unsigned_v<
      decltype(
        std::declval<T>()(
          std::declval<ndextents<extent_of_v<T>>>(),
          std::declval<ndoffsets<extent_of_v<T>>>()
        )
      )
    > > > : std::false_type {};

/**
 * Indicate if a type is a valid extent type.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool is_extent_type_v = is_extent_type<T>::value;

namespace detail {

/**
 * Type traits to create a non-deduced context.
 *
 * @note In C++20 we can simply use `type_identity_t<T>`.
 *
 * @tparam T type
 */
template <typename T>
struct disable_deduction {
  using type = T;
};

/**
 * SFINAE helper to obtain `T` in a non-deduced context.
 *
 * @tparam T type
 */
template <typename T>
using disable_deduction_t = typename disable_deduction<T>::type;

}  // namespace

/**
 * Index computation type representing a leading-dimension first layout.
 *
 * For a matrix this corresponds to a row-major layout.
 *
 * @tparam N Number of dimensions
 */
struct leading_extent {
  /**
   * Return the offset into a buffer given dimensions and an index.
   *
   * @param dims View dimensions
   * @param index View index
   */
  template <std::size_t N>
  constexpr auto operator()(
    const ndextents<N>& dims,
    const detail::disable_deduction_t<ndoffsets<N>>& index) const noexcept
  {
    decltype(N) offset = 0u;
    // for each index
    for (decltype(N) i = 0u; i < N; i++) {
      // row-major offset for the given index
      auto offset_i = index[i];
      // for each row-major stride
      for (auto j = i + 1u; j < N; j++)
        offset_i *= dims[j];
      // update offset
      offset += offset_i;
    }
    return offset;
  }
};

/**
 * Index computation type representing a trailing-dimension first layout.
 *
 * For a matrix this corresponds to a column-major layout.
 *
 * @deprecated This extent type does not work correctly when slicing as we end
 *  up losing the leading dimension multiplier required for strides.
 *
 * @tparam N Number of dimensions
 */
struct trailing_extent {
  /**
   * Return the offset into a buffer given dimensions and an index.
   *
   * @param dim View dimensions
   * @param index View index
   */
  template <std::size_t N>
  constexpr auto operator()(
    const ndextents<N>& dims,
    const detail::disable_deduction_t<ndoffsets<N>>& index) const noexcept
  {
    decltype(N) offset = 0u;
    // for each index
    for (decltype(N) i = 0u; i < N; i++) {
      // col-major offset for the given index
      auto offset_i = index[i];
      // for each col-major stride
      for (decltype(N) j = 0u; j < i; j++)
        offset_i *= dims[j];
      // update offset
      offset += offset_i;
    }
    return offset;
  }
};

/**
 * N-dimensional view type.
 *
 * @tparam E Extent type
 * @tparam T Element type
 * @tparam N Number of dimensions
 */
template <typename E, typename T, std::size_t N>
class ndspan {
public:
  /**
   * Default ctor.
   *
   * Constructs a fully zero-initialized `ndspan`.
   */
  constexpr ndspan() noexcept = default;

  /**
   * Ctor.
   *
   * This uses the extent type provided by the deduction guide.
   *
   * @param data Data buffer
   * @param dims Data dimensions
   */
  constexpr ndspan(T* data, const ndextents<N>& dims) noexcept
    : data_{data}, dims_{dims}
  {}

  /**
   * Ctor.
   *
   * This enables direct deduction of the extent type.
   *
   * @param data Data buffer
   * @param dims Data dimensions
   */
  constexpr ndspan(E, T* data, const ndextents<N>& dims) noexcept
    : ndspan{data, dims}
  {}

  /**
   * Return the data buffer pointer.
   */
  constexpr auto data() const noexcept { return data_; }

  /**
   * Return the total number of elements in the view.
   */
  constexpr auto size() const noexcept
  {
    std::size_t res = 1u;
    for (auto dim : dims_)
      res *= dim;
    return res;
  }

  /**
   * Return the size of the specified dimension.
   *
   * @param i Dimension index
   */
  constexpr auto size(std::size_t i) const noexcept
  {
    return dims_[i];
  }

  /**
   * Return the size of the specified dimension.
   *
   * @tparam I Dimension index
   */
  template <std::size_t I>
  constexpr auto size() const noexcept
  {
    static_assert(I < N, "indexing out of bounds");
    return dims_[I];
  }

  /**
   * Return a `N - 1` dimensional slice of the view or a value reference.
   *
   * Calls to `operator[]` can be chained, e.g. `view[i][j][k]` as necessary.
   *
   * @param i Leading dimension index
   */
  constexpr decltype(auto) operator[](std::size_t i) const noexcept
  {
    // if only 1 dimension then reference element
    if constexpr (N == 1u)
      return data_[i];
    // otherwise, produce another view
    // note: need disambiguator since dims_ is template parameter dependent
    else
      return ndspan<E, T, N - 1u>{
        data_ + E{}(dims_, {i}),    // data + extent offset
        dims_.template slice<1u>()  // sliced dimensions
      };
  }

  /**
   * Directly reference a specified value.
   *
   * @tparam Ts Integral types
   *
   * @param is Index vales
   */
  template <typename... Ts>
  std::enable_if_t<sizeof...(Ts) == N && (std::is_integral_v<Ts> && ...), T&>
  operator()(Ts... is) const noexcept
  {
    return data_[E{}(dims_, {is...})];
  }

private:
  T* data_{};
  ndextents<N> dims_;
};

// deduction guide when extent is not provided
template <typename T, std::size_t N>
ndspan(T*, const ndextents<N>&) -> ndspan<leading_extent, T, N>;

namespace detail {

/**
 * Write the `ndspan` values to an output stream.
 *
 * The output is in the form of a nested list.
 *
 * @tparam E Extent type
 * @tparam T Element type
 * @tparam N Number of dimensions
 */
template <typename E, typename T, std::size_t N>
void write(std::ostream& out, const ndspan<E, T, N>& span)
{
  out << "[";
  for (std::size_t i = 0u; i < span.template size<0u>(); i++) {
    if (i)
      out << ", ";
    // if N == 1 we have to use operator<< for the value
    if constexpr (N == 1u)
      out << span[i];
    // otherwise we recurse back into write()
    else
      write(out, span[i]);
  }
  out << "]";
}

}  // namespace detail

/**
 * Stream the `ndspan` to an output stream.
 *
 * The output is in the form of `ndspan[N]: [[ ... ]]`.
 *
 * @tparam E Extent type
 * @tparam T Element type
 * @tparam N Number of dimensions
 */
template <typename E, typename T, std::size_t N>
auto& operator<<(std::ostream& out, const ndspan<E, T, N>& span)
{
  out << "ndspan[" << N << "]: ";
  detail::write(out, span);
  return out;
}

}  // namespace npygl

#endif  // NPYGL_NDSPAN_HH_
