#!/usr/bin/bash
#
# Test running script for npyglue.
#
# Author: Derek Huang
# Copyright: MIT License
#

# program name ($0 refers to current function)
PROGNAME=$0
# current action to take, argument parsing mode
RUN_ACTION=
PARSE_ACTION=
# CTest arguments
# note: nonempty by default
CTEST_ARGS="-j$(nproc)"
# default build output directory
BUILD_DIR=build

##
# Print build script usage.
#
print_usage() {
    echo "Usage: $PROGNAME [-h] [-t TEST_DIR] [-j[ ]PROCS] [-p] [-Ct CTEST_ARGS]"
    echo
    echo "Testing harness script for npyglue *nix builds."
    echo
    echo "Only supports single-configuration CMake generators, e.g. Makefile"
    echo "generators or Ninja, with \"Unix Makefiles\" as the default."
    echo
    echo "\$(nproc) tests are run in parallel by CTest by default unless the"
    echo "-j, --parallel argument is provided. To print short progress output "
    echo "the -p, --progress option can be specified."
    echo
    echo "Options:"
    echo "  -h,  --help                     Print this usage"
    echo
    echo "  -t TEST_DIR, --test-dir TEST_DIR"
    echo "                                  Test build directory, default" \
        "$BUILD_DIR"
    echo
    echo "  -j[ ]PROCS, --parallel PROCS    Test parallelism, default $(nproc)"
    echo "  -p, --progress                  Print short progress output"
    echo
    echo "  -Ct CTEST_ARGS, --ctest-args CTEST_ARGS"
    echo "                                  Additional CTest arguments"
}

##
# Parse incoming arguments and populate CTest args.
#
# Arguments:
#   List of command-line arguments
#
parse_args() {
    for ARG in $@
    do
        case $ARG in
        # break early to print usage
        -h | --help)
            RUN_ACTION=print_usage
            return 0
            ;;
        # set build directory to test
        -t | --test-dir)
            PARSE_ACTION=test_dir
            ;;
        # CTest parallel level
        -j | --progress)
            PARSE_ACTION=ctest_parallel
            ;;
        # -j[0-9]+ to pass to CTest
        -j*)
            CTEST_ARGS="$CTEST_ARGS $ARG"
            ;;
        # CTest progress output
        -p | --progress)
            CTEST_ARGS="$CTEST_ARGS --progress"
            ;;
        # collect CTest args
        -Ct | --ctest-args)
            PARSE_ACTION=ctest_args
            ;;
        # operate according to PARSE_ACTION
        *)
            case $PARSE_ACTION in
            test_dir)
                BUILD_DIR=$ARG
                ;;
            ctest_parallel)
                CTEST_ARGS="$CTEST_ARGS -j$ARG"
                ;;
            ctest_args)
                CTEST_ARGS="$CTEST_ARGS $ARG"
                ;;
            # no parse action
            *)
                echo "Error: Unknown option '$ARG'." \
                    "Try $PROGNAME --help for usage."
                return 1
            esac
            ;;
        esac
    done
    return 0
}

##
# Main entry point.
#
# Arguments:
#   List of command-line arguments
#
main () {
    # parse args and exit if error
    parse_args "$@"
    if [ $? -ne 0 ]; then return $?; fi
    # handle actions. either print usage or build
    if [ "$RUN_ACTION" = "print_usage" ]
    then
        print_usage
    else
        ctest --test-dir $BUILD_DIR $CTEST_ARGS
    fi
    # propagate last error code
    return $?
}

main "$@"
