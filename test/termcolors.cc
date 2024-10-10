/**
 * @file termcolors.cc
 * @author Derek Huang
 * @brief C++ program that prints SGR values and 8-bit terminal colors
 * @copyright MIT License
 *
 * This is a highly modified C++ version of the original C code from
 * https://en.wikipedia.org/wiki/ANSI_escape_code#In_C that prints:
 *
 * 1. All 108 SGR values
 * 2. All 256 8-bit foreground colors
 * 2. All 256 8-bit background colors
 *
 * This program also works on Windows by ensuring virtual terminal sequences
 * are enabled via a RAII context before any escape sequences are used.
 */

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <type_traits>

#include "npygl/termcolor.hh"

namespace {

/**
 * Write the 108 basic SGR values, with applied formatting, to the stream.
 *
 * @param out Output stream
 */
void sgr_print(std::ostream& out = std::cout)
{
  // row/column width + print field width
  // note: fwidth is int since std::setw takes an int
  constexpr auto width = 10u;
  constexpr auto fwidth = 4;
  // print first 100 SGR values
  for (unsigned i = 0; i < width; i++) {
    for (unsigned j = 0; j < width; j++) {
      // SGR value to print
      // note: even if all the operands were short, arithmetic must be done on
      // int so we would still have to narrow the type back
      auto v = static_cast<unsigned short>(width * i + j);
      // print numerical value with formatting applied
      out << npygl::sgr_value{v} << std::setw(fwidth) << v << npygl::vts::normal;
    }
    out << std::endl;
  }
  // print last 8 SGR values
  for (unsigned short i = 0; i < 8; i++) {
    auto v = static_cast<unsigned short>(width * width + i);
    out << npygl::sgr_value{v} << std::setw(fwidth) << v << npygl::vts::normal;
  }
  out << std::endl;
}

/**
 * Write the 256 8-bit color values with applied formatting.
 *
 * @tparam S Extended SGR type
 *
 * @param out Output stream
 */
template <typename S, typename = std::enable_if_t<npygl::is_sgr_value_ext_v<S>>>
void sgr_ext_print(std::ostream& out = std::cout)
{
  // row/column width + print field width
  // note: fwidth is int since std::setw takes an int
  constexpr auto width = 16u;
  constexpr auto fwidth = 4;
  // print the 8-bit colors
  for (unsigned i = 0; i < width; i++) {
    for (unsigned j = 0; j < width; j++) {
      // extended SGR value to print
      auto v = static_cast<unsigned short>(width * i + j);
      // print numberical value with formatting
      out << S{v} << std::setw(fwidth) << v << npygl::vts::normal;
    }
    out << std::endl;
  }
}

}  // namespace

int main()
{
  npygl::vts_stdout_context ctx;
  // print SGR values
  std::cout << "SGR values:\n";
  sgr_print();
  // print 8-bit foreground colors
  std::cout << "\nForeground colors:\n";
  sgr_ext_print<npygl::sgr_fg_color>();
  // print 8-bit background colors
  std::cout << "\nBackground colors:\n";
  sgr_ext_print<npygl::sgr_bg_color>();
  return EXIT_SUCCESS;
}
