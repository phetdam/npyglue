/**
 * @file colorgrid.cc
 * @author Derek Huang
 * @brief C++ program that prints a 16x16 color grid
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

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif  // _WIN32

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <type_traits>

#include "npygl/features.h"

#if NPYGL_HAS_UNISTD_H
#include <unistd.h>
#endif  // NPYGL_HAS_UNISTD_H

namespace {

/**
 * The sequence prefix used for SGR in-band terminal settings.
 */
constexpr const char* sgr_prefix = "\033[";

/**
 * The sequence suffix used for SGR in-band terminal settings.
 */
constexpr const char* sgr_suffix = "m";

/**
 * A basic SGR value.
 */
class sgr_value {
public:
  /**
   * Ctor.
   *
   * @param value SGR value from 0 through 107
   */
  constexpr sgr_value(unsigned short value) noexcept : value_{value} {}

  /**
   * Return the raw SGR value.
   */
  constexpr auto value() const noexcept { return value_; }

private:
  unsigned short value_;
};

/**
 * Extended SGR value used for setting an 8-bit terminal color.
 *
 * This is for setting the foreground, background, or underline to one of the
 * 256 8-bit colors, which is also stored as a member.
 *
 * @tparam val_ SGR foreground (38), background (48), or underline (58) value
 */
template <unsigned short val_>
class sgr_value_ext : public sgr_value {
public:
  static_assert(val_ == 38 || val_ == 48 || val_ == 58);

  /**
   * Ctor.
   *
   * @param color 8-bit color value to set
   */
  constexpr sgr_value_ext(unsigned short color) noexcept
    : sgr_value{val_}, color_{color}
  {}

  /**
   * Return the 8-bit color that will be used.
   */
  constexpr auto color() const noexcept { return color_; }

private:
  unsigned short color_;
};

/**
 * Concrete types for setting 8-bit foreground, background, underline colors.
 */
using sgr_fg_color = sgr_value_ext<38>;
using sgr_bg_color = sgr_value_ext<48>;
using sgr_ul_color = sgr_value_ext<58>;

/**
 * Traits class to indicate if a type is an extended SGR value type.
 *
 * @tparam T type
 */
template <typename T>
struct is_sgr_value_ext : std::bool_constant<
  std::is_same_v<T, sgr_fg_color> ||
  std::is_same_v<T, sgr_bg_color> ||
  std::is_same_v<T, sgr_ul_color>
> {};

/**
 * Helper to indicate if a type is an extended SGR value type.
 *
 * @tparam T type
 */
template <typename T>
constexpr bool is_sgr_value_ext_v = is_sgr_value_ext<T>::value;

/**
 * Stream a basic SGR value to the output stream.
 *
 * @param out Output stream
 * @param value SGR value
 */
inline auto& operator<<(std::ostream& out, sgr_value value)
{
  return out << sgr_prefix << value.value() << sgr_suffix;
}

/**
 * Stream an extended SGR value to the output stream.
 *
 * @tparam val_ SGR foreground (38), background (48), or underline (58) value
 *
 * @param out Output stream
 * @param value Extended SGR value
 */
template <unsigned short val_>
inline auto& operator<<(std::ostream& out, sgr_value_ext<val_> value)
{
  return out << sgr_prefix << val_ << ";5;" << value.color() << sgr_suffix;
}

namespace vts {

/**
 * Global `constexpr` objects for stream manipulation.
 *
 * @todo Very much incomplete as missing even the basic fg/bg colors.
 */
constexpr sgr_value normal{0};
constexpr sgr_value bright{1};
constexpr sgr_value dimmed{2};
constexpr sgr_value italic{3};
constexpr sgr_value underline{4};
constexpr sgr_value slow_blink{5};
constexpr sgr_value fast_blink{6};
constexpr sgr_value invert{7};
constexpr sgr_value hide{8};
// note: alias provided for convenience
constexpr sgr_value strikeout{9};
constexpr sgr_value strikethrough{9};
constexpr sgr_value primary_font{10};

constexpr sgr_value fg_green{32};

/**
 * Return an extended SGR value to set the foreground color.
 *
 * @param v 8-bit color value
 */
inline sgr_fg_color fg_color(unsigned short v)
{
  return v;
}

/**
 * Return an extended SGR value to set the background color.
 *
 * @param v 8-bit color value
 */
inline sgr_bg_color bg_color(unsigned short v)
{
  return v;
}

}  // namespace vts

/**
 * Enable virtual terminal sequences for standard output.
 *
 * This provides an RAII wrapper to enabling/disabling VTS for standard output
 * on WIndows as ANSI control sequences are disabled by default.
 *
 * On POSIX systems this object is a no-op since control sequences are enabled.
 */
class vts_stdout_context {
public:
  /**
   * Ctor.
   *
   * On Windows, if there is an error, `GetLastError()` should be called.
   */
  vts_stdout_context() noexcept
#if defined(_WIN32)
    : handle_{GetStdHandle(STD_OUTPUT_HANDLE)}
#else
    : handle_{STDOUT_FILENO}
#endif  // !defined(_WIN32)
  {
#ifdef _WIN32
    // error when retrieving handle
    if (handle_ == INVALID_HANDLE_VALUE)
      return;
    // no error, so get console mode
    if (!GetConsoleMode(handle_, &mode_)) {
      handle_ = INVALID_HANDLE_VALUE;
      return;
    }
    // attempt to enable processed output + VTS handling
    mode_ |= (ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    if (!SetConsoleMode(handle_, mode_)) {
      handle_ = INVALID_HANDLE_VALUE;
      return;
    }
#endif  // _WIN32
  }

  /**
   * Dtor.
   */
  ~vts_stdout_context()
  {
#ifdef _WIN32
    // no-op if error
    if (handle_ == INVALID_HANDLE_VALUE)
      return;
    // otherwise, unset the flags
    mode_ &= (~ENABLE_PROCESSED_OUTPUT & ~ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    // ignore return; nothing else to do
    SetConsoleMode(handle_, mode_);
#endif  // _WIN32
  }

  /**
   * Return the standard output handle.
   *
   * On Windows if there is an error this returns `INVALID_HANDLE_VALUE`.
   */
  auto handle() const noexcept { return handle_; }

private:
#if defined(_WIN32)
  HANDLE handle_;
  DWORD mode_{};
#else
  int handle_;
  int mode_{};
#endif  // !defined(_WIN32)
};

// some examples of how we would like to use the terminal color library:
//
// std::cout << npygl::vts::bright << "hello world" << npygl::vts::normal;
// std::cout << npygl::vts::fg_green << "bar" << npygl::vts::normal;
// std::cout << npygl::vts::fg_color(128) << "foo" << npygl::vts::normal;

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
      out << sgr_value{v} << std::setw(fwidth) << v << vts::normal;
    }
    out << std::endl;
  }
  // print last 8 SGR values
  for (unsigned short i = 0; i < 8; i++) {
    auto v = static_cast<unsigned short>(width * width + i);
    out << sgr_value{v} << std::setw(fwidth) << v << vts::normal;
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
template <typename S, typename = std::enable_if_t<is_sgr_value_ext_v<S>>>
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
      out << S{v} << std::setw(fwidth) << v << vts::normal;
    }
    out << std::endl;
  }
}

}  // namespace

int main()
{
  vts_stdout_context ctx;
  // print SGR values
  std::cout << "SGR values:\n";
  sgr_print();
  // print 8-bit foreground colors
  std::cout << "\nForeground colors:\n";
  sgr_ext_print<sgr_fg_color>();
  // print 8-bit background colors
  std::cout << "\nBackground colors:\n";
  sgr_ext_print<sgr_bg_color>();
  return EXIT_SUCCESS;
}
