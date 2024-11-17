/**
 * @file testing/ostream.hh
 * @author Derek Huang
 * @brief C++ header for ostream.hh testing support
 * @copyright MIT License
 */

#ifndef NPYGL_TESTING_OSTREAM_HH_
#define NPYGL_TESTING_OSTREAM_HH_

#include <ostream>

#include "npygl/demangle.hh"

namespace npygl {
namespace testing {

/**
 * Trivial type that does not have an insertion operator defined.
 */
struct not_ostreamable_type {};

/**
 * Trivial type that *does* have an insertion operator defined.
 */
struct ostreamable_type {};

/**
 * Stream insertion operator for the `ostreamable_type.
 */
inline auto& operator<<(std::ostream& out, ostreamable_type /*value*/)
{
  return out << type_name(typeid(ostreamable_type));
}

}  // namespace testing
}  // namespace npygl

#endif  // NPYGL_TESTING_OSTREAM_HH_
