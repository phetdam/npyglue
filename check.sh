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
CTEST_ARGS=
# default build output directory and build configuration (unused)
BUILD_DIR=build
# BUILD_CONFIG=Debug

##
# Print build script usage.
#
print_usage() {
    echo "Usage: $PROGNAME [-h] [-t TEST_DIR] [-Ct CTEST_ARGS]"
    echo
    echo "Testing harness script for npyglue *nix builds."
    echo
    echo "Only supports single-configuration CMake generators, e.g. Makefile"
    echo "generators or Ninja, with \"Unix Makefiles\" as the default."
    echo
    echo "Options:"
    echo "  -h,  --help                     Print this usage"
    echo "  -t,  --test-dir TEST_DIR        Build directory to test, default" \
        "$BUILD_DIR"
    # echo "  -c,  --config CONFIG            Build configuration, default" \
    #     "$BUILD_CONFIG"
    echo "  -Ct, --ctest-args CTEST_ARGS    Args to pass to ctest test command"
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
            # set build output directory
            -o | --output-dir)
                PARSE_ACTION=output_dir
                ;;
            # set build configuration
            # -c | --config)
            #     PARSE_ACTION=build_config
            #     ;;
            # collect CTest args
            -Ct | --ctest-args)
                PARSE_ACTION=ctest_args
                ;;
            # operate according to PARSE_ACTION
            *)
                # set build output dir
                if [ "$PARSE_ACTION" = output_dir ]
                then
                    BUILD_DIR=$ARG
                # set build configuration
                # elif [ "$PARSE_ACTION" = build_config ]
                # then
                #     BUILD_CONFIG=$ARG
                # update CTest args
                elif [ "$PARSE_ACTION" = ctest_args ]
                then
                    # assign directly if empty to prevent adding extra space
                    if [ -z "$CTEST_ARGS" ]
                    then
                        CTEST_ARGS=$ARG
                    else
                        CTEST_ARGS="$CTEST_ARGS $ARG"
                    fi
                # error otherwise
                else
                    echo "Error: Unknown option '$ARG'." \
                        "Try $PROGNAME --help for usage."
                    return 1
                fi
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
        # TODO: if we add per-config subdirectories during build, just need to
        # append BUILD_CONFIG to the end of BUILD_DIR
        ctest --test-dir $BUILD_DIR -j$(nproc) $CTEST_ARGS
    fi
    return 0
}

main "$@"
