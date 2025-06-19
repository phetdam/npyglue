/**
 * @file traits_checker_filter_test.cc
 * @author Derek Huang
 * @brief C++ program testing traits_checker_driver test filtering features
 * @copyright MIT License
 */

#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include "npygl/ctti.hh"
#include "npygl/demangle.hh"
#include "npygl/termcolor.hh"
#include "npygl/testing/traits_checker.hh"
#include "npygl/type_traits.hh"
#include "npygl/version.h"

namespace {

// base test driver type. this will be subclassed to implement the experimental
// selective test filtering/execution features
// FIXME: somehow one test is not listed when using type_traits_test_driver
using driver_type = npygl::testing::traits_checker_driver<
  npygl::testing::traits_checker<
    npygl::has_static_size,
    std::tuple<
      std::tuple<int, double, char>,
      std::pair<std::vector<double>, std::false_type>,
      std::pair<std::tuple<char, unsigned, std::string>, std::true_type>,
      std::pair<double, const volatile void*>,
      std::array<unsigned, 100>,
      char[256],
      std::pair<std::string, std::false_type>,
      std::pair<std::map<std::string, unsigned>, std::false_type>,
      int[2][3][4],
      std::pair<double[], std::false_type>,
      npygl::testing::skipped<double>,
      npygl::testing::skipped<std::pair<void**, std::false_type>>
    >
  >,
  npygl::testing::traits_checker<
    npygl::is_monomorphic_tuple,
    std::tuple<
      std::tuple<int, int, int>,
      std::pair<std::tuple<double, char>, std::false_type>,
      std::pair<std::tuple<char*, char*, char*>, std::true_type>,
      std::pair<std::tuple<void*, char**, double>, std::false_type>,
      npygl::testing::skipped<std::tuple<int, char, const char*>>
    >
  >
>;

/**
 * Concrete test driver derived type for experimenting with test filtering.
 *
 * This is treated as if a `traits_checker_driver<Ts...>` was being augmented
 * with additional member functions. Therefore, we operate as though either a
 * `Ts...` pack of `traits_checker<Traits, T>` specializations is available,
 * where `Ts` might be one type and so `Traits` and `T` are available.
 */
struct filter_test_driver : public driver_type {
  // prevent shadowing
  using driver_type::operator();

  /**
   * Runtime test map for test filtering.
   *
   * This maps the test name, which is obtained via invocation of a
   * `traits_checker_case_formatter<traits_checker_case_t<Traits, T>>`.
   *
   * @todo `std::function` specialization should take a `std::ostream&`.
   *
   * @note The mapping is by test name and is not in order of test definition.
   */
  using runtime_test_map = std::unordered_map<std::string, std::function<bool()>>;

  /**
   * Run the traits checker test suite and indicate success or failure.
   *
   * This overload takes arguments from `main()` to allow test filtering and
   * should be used in `main()` to run the entire test program.
   *
   * @param argc Argument count from `main()`
   * @param argv Argument vector from `main()`
   * @return `true` on success, `false` on failure
   */
  bool operator()(int argc, char** argv) const;
};

// mapping of name -> test executor
// note: temporary
using test_mapping = filter_test_driver::runtime_test_map;

/**
 * Creates a runtime test mapping for a `traits_checker<Traits, T>`.
 *
 * This defines a callable that returns the runtime test mapping for the test
 * case(s) that are defined by the `traits_checker<Traits, T>`. All tests are
 * included, even ones that are marked as skipped.
 *
 * @tparam T `traits_checker<Traits, Input>` specialization
 */
template <typename T, typename = void>
struct traits_checker_runtime_mapper {};

/**
 * Partial specialization for a `traits_checker<Traits, std::tuple<Ts...>>`.
 *
 * @tparam Traits Unary traits type with a `value` member
 * @tparam Ts Input types, e.g. `std::pair<Input, compare_or_truth_type>`
 */
template <template <typename> typename Traits, typename... Ts>
struct traits_checker_runtime_mapper<
  npygl::testing::traits_checker<Traits, std::tuple<Ts...>>, void /*unused*/> {
  /**
   * Return the runtime test mapping for this `traits_checker` specialization.
   */
  auto operator()() const
  {
    test_mapping map;
    // note: duplicate input types (which really should not occur) ignored
    (
      [&map]
      {
        using namespace npygl::testing;
        // actual traits_type, compare_or_truth_type test case type pair. if Ts
        // is a skipped<T>, we need to unwrap to get the T before formatting
        using test_case = traits_checker_case_t<Traits, remove_skipped_t<Ts>>;
        // get test case name using formatter
        auto name = []
        {
          std::stringstream ss;
          traits_checker_case_formatter<test_case>{}(ss);
          return ss.str();
        }();
        // insert wrapped test callable. we wrap the Ts input type in an extra
        // tuple to prevent splitting into multiple test cases
        map[std::move(name)] = []
        {
          return traits_checker<Traits, std::tuple<Ts>>{}();
        };
      }(), ...
    );
    return map;
  }
};

/**
 * Partial specialization for a `traits_checker<Traits, T>` with non-tuple `T`.
 *
 * This is secondary to the `std::tuple<Ts...>` specialization because in order
 * to handle tuple inputs in a tuple of inputs we need to "unwrap" one tuple.
 *
 * @tparam Traits Unary traits type with a `value` member
 * @tparam T Input type, e.g. `std::pair<Input, compare_or_truth_type>`
 */
template <template <typename> typename Traits, typename T>
struct traits_checker_runtime_mapper<
  npygl::testing::traits_checker<Traits, T>,
  std::enable_if_t<!npygl::is_tuple_v<T>> > : traits_checker_runtime_mapper<
    npygl::testing::traits_checker<Traits, std::tuple<T>>> {};

/**
 * Creates a runtime test mapping for a `traits_checker_driver<Ts...>`.
 *
 * This defines a callable that returns the runtime test mapping for the test
 * case(s) that are defined by all the `traits_checker<Traits, T>`.
 *
 * @tparam T `traits_checker_driver<Ts...>` specialization
 */
template <typename T>
struct traits_checker_driver_runtime_mapper {};

/**
 * Partial specialization for a `traits_checker_driver<Ts...>`.
 *
 * @tparam Ts `traits_checker<Traits, T>` specializations
 */
template <typename... Ts>
struct traits_checker_driver_runtime_mapper<
  npygl::testing::traits_checker_driver<Ts...> > {
  /**
   * Return the runtime test mapping for the `traits_checker_driver<Ts...>`.
   */
  auto operator()() const
  {
    using namespace npygl::testing;
    test_mapping map;
    // construct from each traits_checker<Traits, T>
    (map.merge(traits_checker_runtime_mapper<Ts>{}()), ...);
    return map;
  }
};

////////////////////////////////////////////////////////////////////////////////
// operator() implementation for the filter_test_driver                       //
////////////////////////////////////////////////////////////////////////////////

/**
 * Enum indicating the action the `filter_test_driver::operator()` should take.
 */
enum class traits_checker_run_action {
  /** print the driver help */
  print_usage,
  /** print version info */
  print_version,
  /** list the registered test cases */
  list_tests,
  /** run a verbatim single test name */
  run_test,
  /** run tests matching the given regex pattern */
  run_test_pattern,
  /** run all the tests (default) */
  run_all_tests,
  /** print the driver type name (can be very long) */
  print_driver_type
};

/**
 * Command-line options accepted by the `traits_checker_driver<Ts...>`.
 *
 * @param action Action to take; runs all tests by default
 * @param test_pattern Test name or pattern to run depending on the action
 */
struct traits_checker_options {
  traits_checker_run_action action = traits_checker_run_action::run_all_tests;
  std::string test_pattern;
};

/**
 * Parse command-line options for the `traits_checker_driver<Ts...>`.
 *
 * @param opts Options to populate
 * @param argc Argument count from `main()`
 * @param argv Argument vector from `main()`
 * @returns `true` if parsing succeeded, `false` otherwise
 */
bool parse_args(traits_checker_options& opts, int argc, char** argv)
{
  for (int i = 1; i < argc; i++) {
    // use string_view for convenience
    std::string_view arg{argv[i]};
    // -h, --help. override + short circuit if printing help
    if (arg == "-h" || arg == "--help") {
      opts.action = traits_checker_run_action::print_usage;
      return true;
    }
    // -v, --version
    else if (arg == "-v" || arg == "--version")
      opts.action = traits_checker_run_action::print_version;
    // -l, --list-tests
    else if (arg == "-l" || arg == "--list-tests")
      opts.action = traits_checker_run_action::list_tests;
    // -t, --test-name
    else if (arg == "-t" || arg == "--test-name") {
      // no argument
      if (++i == argc) {
        std::cerr << "Error: No argument provided for -t, --test-name" <<
          std::endl;
        return false;
      }
      // use test name
      opts.action = traits_checker_run_action::run_test;
      opts.test_pattern = argv[i];
    }
    // -T, --test-pattern
    else if (arg == "-T" || arg == "--test-pattern") {
      // no argument
      if (++i == argc) {
        std::cerr << "Error: No argument provided for -T, --test-pattern" <<
          std::endl;
        return false;
      }
      // use test pattern
      opts.action = traits_checker_run_action::run_test_pattern;
      opts.test_pattern = argv[i];
    }
    // --print-driver-type
    else if (arg == "--print-driver-type")
      opts.action = traits_checker_run_action::print_driver_type;
    // unknown option
    else {
      std::cerr << "Error: Unknown option \"" << arg <<
        "\". Try -h, --help for usage" << std::endl;
      return false;
    }
  }
  // done
  return true;
}

// TODO: organize
bool
filter_test_driver::operator()(int argc, char** argv) const
{
  // parse incoming args
  traits_checker_options opts;
  if (!parse_args(opts, argc, argv))
    return false;
  // test mapper type alias
  // note: cheap to create as it had no data members
  // TODO: allow subclasses of the traits_checker_driver<Ts...>?
  traits_checker_driver_runtime_mapper<driver_type> test_mapper;
  // switch off action
  // note: all cases are bracketed to code folding in editors
  switch (opts.action) {
    // print usage
    case traits_checker_run_action::print_usage: {
      // TODO: stub for now
      std::cout <<
"Usage: <progname> [-h] [-v |\n"
"                        -l |\n"
"                        -t TEST_NAME |\n"
"                        -T TEST_PATTERN |\n"
"                        --print-driver-type]\n"
"\n"
"Run the tests registered with the type traits checker driver.\n"
"\n"
"If multiple options are specified the last option is considered. -h, --help\n"
"will override all other options presented. If no options are provided, then\n"
"all the type traits tests will be run as usual.\n"
"\n"
"Options:\n"
"  -h, --help             Print this usage\n"
"  -v, --version          Print npyglue version information\n"
"  -l, --list-tests       List all traits checker test names\n"
"\n"
"  -t TEST_NAME, --test-name TEST_NAME\n"
"                         Run the specified traits checker test\n"
"\n"
"  -T TEST_PATTERN, --test-pattern TEST_PATTERN\n"
"                         Run all traits checker tests matching the given regex\n"
"                         pattern. Special characters may need to be escaped.\n"
"\n"
"  --print-driver-type    Print the traits_checker_driver<Ts...> demangled type.\n"
"                         Be warned that the type name may be very long.\n";
      std::cout << std::flush;
      return true;
    }
    // print version info
    case traits_checker_run_action::print_version: {
      std::cout << "npyglue " NPYGL_VERSION << std::endl;
      return true;
    }
    // list all tests
    case traits_checker_run_action::list_tests: {
      // print test names
      for (const auto& [name, _] : test_mapper())
        std::cout << name << '\n';
      // flush and return
      std::cout << std::flush;
      return true;
    }
    // print driver type
    // note: using type_name<T>() since abi::__cxa_demangle fails on long types
    case traits_checker_run_action::print_driver_type: {
      std::cout << npygl::type_name<driver_type>() << std::endl;
      return true;
    }
    // run single test
    case traits_checker_run_action::run_test: {
      // attempt to locate test
      auto test_map = test_mapper();
      // not found
      if (test_map.find(opts.test_pattern) == test_map.end()) {
        std::cerr << "Error: No test with name " << opts.test_pattern <<
          " found" << std::endl;
        return false;
      }
      // otherwise, run single test
      return test_map[opts.test_pattern]();
    }
    // run tests matching regex pattern
    case traits_checker_run_action::run_test_pattern: {
      // regex pattern for search
      std::regex pattern{opts.test_pattern};
      // popluate vector of tests to run
      std::vector<typename test_mapping::mapped_type> tests;
      for (const auto& [name, test] : test_mapper())
        if (std::regex_search(name, pattern))
          tests.push_back(test);
      // TODO: should we error if nothing is matched?
      // execute selected tests
      auto n_failed = 0u;
      for (const auto& test : tests)
        n_failed += !test();
      // done
      return !n_failed;
    }
    // default (run_all_tests) is to run all the tests
    default:
      return operator()();
  }
}

// experimental test driver
constexpr filter_test_driver driver;

}  // namespace

int main(int argc, char** argv)
{
  // note: no-op on POSIX systems and unnecessary in Windows Terminal
  npygl::vts_stdout_context ctx;
  // run with command-line options
  return driver(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
