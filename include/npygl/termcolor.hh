/**
 * @file termcolor.hh
 * @author Derek Huang
 * @brief C++ header for output stream formatting via ANSI control sequences
 * @copyright MIT License
 *
 * @note On Windows linking against `kernel32` is required.
 */

#ifndef NPYGL_TERMCOLOR_HH_
#define NPYGL_TERMCOLOR_HH_

#ifdef _WIN32
// reduce Windows.h include size
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif  // _WIN32

#include <ostream>
#include <type_traits>

#include "npygl/features.h"

// for STDOUT_FILENO
#if NPYGL_HAS_UNISTD_H
#include <unistd.h>
#endif  // NPYGL_HAS_UNISTD_H

namespace npygl {

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

/**
 * Global `constexpr` objects for stream manipulation.
 *
 * Some objects provide alias for the same SGR values. See the Microsoft
 * documentation on virtual terminal sequences and the ANSI escape codes
 * Wikipedia page for more SGR values that may not be listed here.
 */
namespace vts {

////////////////////////////////////////////////////////////////////////////////
// text formatting
////////////////////////////////////////////////////////////////////////////////
constexpr sgr_value normal{0};
constexpr sgr_value bright{1};
constexpr sgr_value dimmed{2};
constexpr sgr_value italic{3};
constexpr sgr_value underline{4};
constexpr sgr_value slow_blink{5};
constexpr sgr_value fast_blink{6};
constexpr sgr_value invert{7};
// note: alias provided for convenience
constexpr sgr_value hide{8};
constexpr sgr_value conceal{8};
// note: alias provided for convenience
constexpr sgr_value strikeout{9};
constexpr sgr_value strikethrough{9};
constexpr sgr_value no_bright{22};
constexpr sgr_value no_underline{24};
constexpr sgr_value no_blink{24};
constexpr sgr_value no_invert{27};
// note: alias provided for convenience
constexpr sgr_value unhide{28};
constexpr sgr_value reveal{28};
// note: alias provided for convenience
constexpr sgr_value no_strikeout{29};
constexpr sgr_value no_strikethrough{29};

////////////////////////////////////////////////////////////////////////////////
// basic colors
////////////////////////////////////////////////////////////////////////////////
constexpr sgr_value fg_black{30};
constexpr sgr_value fg_red{31};
constexpr sgr_value fg_green{32};
constexpr sgr_value fg_yellow{33};
constexpr sgr_value fg_blue{34};
constexpr sgr_value fg_magenta{35};
constexpr sgr_value fg_cyan{36};
constexpr sgr_value fg_white{37};
constexpr sgr_value fg_normal{39};
constexpr sgr_value bg_black{40};
constexpr sgr_value bg_red{41};
constexpr sgr_value bg_green{42};
constexpr sgr_value bg_yellow{43};
constexpr sgr_value bg_blue{44};
constexpr sgr_value bg_magenta{45};
constexpr sgr_value bg_cyan{46};
constexpr sgr_value bg_white{47};
constexpr sgr_value bg_normal{49};

////////////////////////////////////////////////////////////////////////////////
// bright basic colors
////////////////////////////////////////////////////////////////////////////////
constexpr sgr_value fg_black_bright{90};
constexpr sgr_value fg_red_bright{91};
constexpr sgr_value fg_green_bright{92};
constexpr sgr_value fg_yellow_bright{93};
constexpr sgr_value fg_blue_bright{94};
constexpr sgr_value fg_magenta_bright{95};
constexpr sgr_value fg_cyan_bright{96};
constexpr sgr_value fg_white_bright{97};
constexpr sgr_value bg_black_bright{100};
constexpr sgr_value bg_red_bright{101};
constexpr sgr_value bg_green_bright{102};
constexpr sgr_value bg_yellow_bright{103};
constexpr sgr_value bg_blue_bright{104};
constexpr sgr_value bg_magenta_bright{105};
constexpr sgr_value bg_cyan_bright{106};
constexpr sgr_value bg_white_bright{107};

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

}  // namespace npygl

#endif  // NPYGL_TERMCOLOR_HH_
