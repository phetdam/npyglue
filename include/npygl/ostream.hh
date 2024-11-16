/**
 * @file ostream.hh
 * @author Derek Huang
 * @brief C++ header for an output stream wrapper allowing << for any object
 * @copyright MIT License
 */

#ifndef NPYGL_OSTREAM_HH_
#define NPYFL_OSTREAM_HH_

#include <iostream>
#include <ostream>

#include "npygl/demangle.hh"
#include "npygl/features.h"
#include "npygl/type_traits.hh"

namespace npygl {

/**
 * Output stream wrapper that enables any type to be streamed.
 *
 * @note The choice to use a stream wrapper over an object wrapper was made in
 *  order to make drop-in replacement of `std::cout` effortless.
 */
class ostream_wrapper {
public:
  /**
   * Ctor.
   *
   * @param out Output stream reference
   */
  ostream_wrapper(std::ostream& out) noexcept : out_{out} {}

  /**
   * Insert a value into the underlying stream.
   *
   * If the data type has its own `operator<<` overload available then it will
   * be streamed as usual to the underlying stream. However, types that would
   * not ordinarily be insertable into a `std::ostream` will have a default
   * representation streamed that includes the object type and address.
   *
   * @tparam T type
   *
   * @param obj Object to stream
   */
  template <typename T>
  auto& operator<<(const T& obj)
  {
    // if streamable, use its own operator<<
    if constexpr (is_ostreamable_v<T>)
      out_ << obj;
    // otherwise, print "default" representation
    else
      out_ << "<object " <<
#if NPYGL_HAS_RTTI
        type_name(typeid(T)) <<
#else
        "(unknown type)" <<
#endif  // !NPYGL_HAS_RTTI
        " at " << &obj << '>';
    return *this;
  }

  /**
   * Call an I/O manipulator on the underlying stream.
   *
   * This allows `std::endl`, `std::flush`, etc. to work correctly.
   *
   * @param func I/O manipulator function
   */
  auto& operator<<(std::ostream& (*func)(std::ostream&))
  {
    func(out_);
    return *this;
  }

private:
  std::ostream& out_;
};

// drop-in replacements for standard output streams
inline ostream_wrapper cout{std::cout};
inline ostream_wrapper cerr{std::cerr};

}  // namespace npygl

#endif  // NPYFL_OSTREAM_HH_
