/**
 * @file py_helpers_test.cc
 * @author Derek Huang
 * @brief C++ program to test the npygl Python helpers
 * @copyright MIT License
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <iostream>

#include "npygl/py_helpers.hh"

namespace {

/**
 * Create and print a few scalars using the `py_object` ctor overloads.
 */
void test_scalars()
{
  // we create object first and call py_error_exit in order to catch errors
  {
    npygl::py_object v{4.33112f};
    npygl::py_error_exit();
    std::cout << "float -> float: " << v << std::endl;
  }
  {
    npygl::py_object v{1.31323};
    npygl::py_error_exit();
    std::cout << "double -> float: " << v << std::endl;
  }
  {
    npygl::py_object v{true};
    npygl::py_error_exit();
    std::cout << "true -> bool: " << v << std::endl;
  }
  {
    npygl::py_object v{false};
    npygl::py_error_exit();
    std::cout << "false -> bool: " << v << std::endl;
  }
  {
    npygl::py_object v{-45};
    npygl::py_error_exit();
    std::cout << "int -> int: " << v << std::endl;
  }
  {
    npygl::py_object v{45U};
    npygl::py_error_exit();
    std::cout << "unsigned int -> int: " << v << std::endl;
  }
  {
    npygl::py_object v{{4.3, 1.22}};
    npygl::py_error_exit();
    std::cout << "Py_complex -> complex: " << v << std::endl;
  }
}

}  // namespace

int main()
{
  // initialize Python + print version
  npygl::py_instance python;
  // note: Py_GetVersion can be called before init
  std::cout << Py_GetVersion() << std::endl;
  // import the math module
  constexpr auto mod_name = "math";
  auto mod = npygl::py_import(mod_name);
  npygl::py_error_exit();
  // get the e value from the module
  constexpr auto ename = "e";
  auto e = npygl::py_getattr(mod, ename);
  npygl::py_error_exit();
  // get the log function from the module
  constexpr auto fname = "log";
  auto f = npygl::py_getattr(mod, fname);
  npygl::py_error_exit();
  // check if callable (it should be)
  npygl::py_error_exit(
    !PyCallable_Check(f),
    PyExc_RuntimeError,
    mod_name, " attribute ", fname, " is not callable"
  );
  // call f on e and get result (should be a double)
  auto res = npygl::py_call_one(f, e);
  npygl::py_error_exit();
  npygl::py_error_exit(!PyFloat_Check(res), PyExc_TypeError, "result not a float");
  // print result
  std::cout << mod_name << "." << fname << "(" << mod_name << "." << ename <<
    ") = " << PyFloat_AS_DOUBLE(res.ref()) << std::endl;
  // check reference count after using py_object to increment count
  {
    std::cout << "res ref count: " << Py_REFCNT(res) << std::endl;
    npygl::py_object r{res, npygl::py_object::incref};
    std::cout << "res ref count: " << Py_REFCNT(res) << std::endl;
  }
  // check reference count again
  std::cout << "res ref count: " << Py_REFCNT(res) << std::endl;
  // test scalars
  test_scalars();
  return EXIT_SUCCESS;
}
