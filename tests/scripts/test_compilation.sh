#!/usr/bin/env bash
# Checks that the implementation compiles, provides every class/function with the predefined
# signature, defines all of them, and that the application builds.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

parse_common_args "Usage: $0 [options]

Runs the compilation / interface check (test_compilation)." "$@"

print_setup
configure
echo

# keep going past failing files, so that all problems are reported at once
# shellcheck disable=SC2046
if cmake --build "$BUILD_DIR" --target test_compilation -j "$JOBS" -- $(keep_going_flag); then
    echo
    echo "test_compilation: PASSED"
else
    echo
    echo "test_compilation: FAILED (see the compiler errors above)"
    exit 1
fi
