#!/usr/bin/env bash
# Builds and runs the unit tests (all, or the selected ones).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

parse_common_args "Usage: $0 [options] [TEST...]

Runs the 14 unit tests, or only the given ones. A TEST is a number (9), a test name
(t05_layer_forward) or a part of it (layer).

Examples:
  $0                     all unit tests against src/
  $0 --solution 1 2 3    tests 1-3 against src_sol/
  $0 -v network          all Network tests, with full output of failures" "$@"

# Build a ctest regex from the selected tests.
filter=()
for test in "${POSITIONAL[@]+"${POSITIONAL[@]}"}"; do
    if [[ "$test" =~ ^[0-9]+$ ]]; then
        filter+=("$(printf 't%02d_' "$((10#$test))")")
    else
        filter+=("$test")
    fi
done
regex=""
if [[ ${#filter[@]} -gt 0 ]]; then
    regex="$(IFS='|'; echo "${filter[*]}")"
fi

print_setup
if [[ -n "$REFERENCE" ]]; then
    echo "Reference:                 $REFERENCE"
else
    if [[ ! -d "$SOLUTION_SRC_DIR" || ! -d "$SOLUTION_INCLUDE_DIR" ]]; then
        echo "error: no reference: neither $PREBUILT_REFERENCE nor $SOLUTION_SRC_DIR exists (see --reference)" >&2
        exit 2
    fi
    echo "Reference:                 built from $SOLUTION_SRC_DIR"
fi
# the reference directories are passed explicitly, so a build directory configured before still uses src_sol/
configure -DDNN_BUILD_TESTS=ON -DDNN_REF_PREBUILT="$REFERENCE" -DDNN_REF_SRC_DIR="$SOLUTION_SRC_DIR" \
          -DDNN_REF_INCLUDE_DIR="$SOLUTION_INCLUDE_DIR"

ctest_args=(--test-dir "$BUILD_DIR" -L unit)
if [[ -n "$regex" ]]; then
    ctest_args+=(-R "$regex")
fi

# Pre-build the selected test executables in parallel. Failures are ignored here: ctest builds a
# failing test again and reports it as failed.
mapfile -t targets < <(ctest "${ctest_args[@]}" -N | sed -n 's/^ *Test *#[0-9]*: //p')
if [[ ${#targets[@]} -eq 0 ]]; then
    echo "error: no unit test matches: ${POSITIONAL[*]}" >&2
    exit 2
fi
echo
echo "Building ${#targets[@]} test(s)..."
# shellcheck disable=SC2046
cmake --build "$BUILD_DIR" -j "$JOBS" --target "${targets[@]}" -- $(keep_going_flag) > "$BUILD_DIR/build.log" 2>&1 || true

echo
if [[ $VERBOSE -eq 1 ]]; then
    ctest_args+=(--output-on-failure)
fi
if ctest "${ctest_args[@]}"; then
    exit 0
fi
if [[ $VERBOSE -eq 0 ]]; then
    echo "Details (compiler errors, differing values): rerun with -v"
fi
exit 1
