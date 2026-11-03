#!/usr/bin/env bash
# Runs the compilation / interface check and all unit tests.

script_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$script_dir/common.sh"

parse_common_args "Usage: $0 [options]

Runs test_compilation and all 14 unit tests." "$@"

compilation=PASSED
unit=PASSED

echo "=== Compilation / interface check ==="
"$script_dir/test_compilation.sh" "$@" || compilation=FAILED

echo
echo "=== Unit tests ==="
"$script_dir/run_unit_tests.sh" "$@" || unit=FAILED

echo
echo "=== Summary ==="
echo "test_compilation: $compilation"
echo "unit tests:       $unit"
[[ $compilation == PASSED && $unit == PASSED ]]
