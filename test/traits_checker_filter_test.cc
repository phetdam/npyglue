/**
 * @file traits_checker_filter_test.cc
 * @author Derek Huang
 * @brief C++ program testing traits_checker_driver test filtering features
 * @copyright MIT License
 */

#include <cstdlib>
#include <functional>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#include "npygl/termcolor.hh"
#include "npygl/demangle.hh"
#include "npygl/testing/traits_checker.hh"
#include "npygl/testing/type_traits_test_driver.hh"
#include "npygl/type_traits.hh"

namespace {

// base test driver type. this will be subclassed to implement the experimental
// selective test filtering/execution features
// TODO: use type_traits_test_driver later for soak testing
// constexpr npygl::testing::type_traits_test_driver driver;
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
      std::pair<double[], std::false_type>
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
 * case(s) that are defined by the `traits_checker<Traits, T>`.
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
        // actual traits_type, compare_or_truth_type test case type pair
        using test_case = traits_checker_case_t<Traits, Ts>;
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
 * Command-line options accepted by the `traits_checker_driver<Ts...>`.
 *
 * @param print_usage Print the driver help
 * @param list_tests List all the test cases to be run
 * @param test_pattern Test pattern to evaluate (run in alphabetical order)
 * @param print_driver_type Print the `traits_checker_driver<Ts...>` type
 */
struct traits_checker_options {
  // TODO: use action enum instead of so many boolean flags
  bool print_usage = false;
  bool list_tests = false;
  std::string test_pattern;
  bool print_driver_type = false;
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
    // -h, --help. short circuit if printing help
    if (arg == "-h" || arg == "--help") {
      opts.print_usage = true;
      return true;
    }
    // -l, --list-tests
    else if (arg == "-l" || arg == "--list-tests")
      opts.list_tests = true;
    // -t, --test-pattern
    else if (arg == "-t" || arg == "--test-pattern") {
      // no argument
      if (++i == argc) {
        std::cerr << "Error: No argument provided for -t, --test-pattern" <<
          std::endl;
        return false;
      }
      // use test pattern
      opts.test_pattern = argv[i];
    }
    // --print-driver-type
    else if (arg == "--print-driver-type")
      opts.print_driver_type = true;
    // unknown option
    else {
      std::cerr << "Error: Unknown option \"" << arg <<
        "\". Try -h, --help for usage" << std::endl;
      return false;
    }
  }
  // flags are nutually exclusive
  if (opts.list_tests + opts.print_driver_type + !opts.test_pattern.empty() > 1) {
    std::cerr << "Error: Only one of  -l, --list-tests, --print-driver-type, "
      " -t, --test-pattern may be specified" << std::endl;
    return false;
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
  // print usage
  // TODO: stub for now
  if (opts.print_usage) {
    std::cout <<
      "Usage: <progname> [-h] [-l | -t TEST_PATTERN | --print-driver-type]" <<
      std::endl;
    return true;
  }
  // test mapper type alias
  // note: cheap to create as it had no data members
  // TODO: allow subclasses of the traits_checker_driver<Ts...>?
  traits_checker_driver_runtime_mapper<driver_type> test_mapper;
  // list tests
  if (opts.list_tests) {
    for (const auto& [name, _] : test_mapper())
      std::cout << name << '\n';
    // flush and return
    std::cout << std::flush;
    return true;
  }
  // print driver type
  // TODO: use type_name<T>() instead? abi::__cxa_demangle fails on long types
  if (opts.print_driver_type) {
    std::cout << npygl::type_name(typeid(driver_type)) << std::endl;
    return true;
  }
  // run test patterns
  if (opts.test_pattern.size()) {
    // regex pattern for search
    std::regex pattern{opts.test_pattern};
    // popluate vector of tests to run
    std::vector<typename test_mapping::mapped_type> tests;
    for (const auto& [name, test] : test_mapper())
      if (std::regex_search(name, pattern))
        tests.push_back(test);
    // execute selected tests
    auto n_failed = 0u;
    for (const auto& test : tests)
      n_failed += !test();
    // done
    return !n_failed;
  }
  // no special options so run all test cases
  return operator()();
}

// experimenal test driver
constexpr filter_test_driver driver;

}  // namespace

int main(int argc, char** argv)
{
  // note: no-op on POSIX systems and unnecessary in Windows Terminal
  npygl::vts_stdout_context ctx;
  // run with command-line options
  return driver(argc, argv);
}
