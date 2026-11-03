# Shared option parsing and helpers for the test scripts. Sourced, not executed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREBUILT_REFERENCE="$REPO_ROOT/tests/reference/prebuilt/libdnn_ref.a"

# implementation under test (the students' version) and full implementation (reference)
SRC_DIR="$REPO_ROOT/src"
INCLUDE_DIR="$REPO_ROOT/include"
SOLUTION_SRC_DIR="$REPO_ROOT/src_sol"
SOLUTION_INCLUDE_DIR="$REPO_ROOT/include_sol"
BUILD_DIR="$REPO_ROOT/build-tests"
REFERENCE=""
REFERENCE_FROM_SOURCE=0
JOBS="$(nproc 2>/dev/null || echo 4)"
VERBOSE=0
POSITIONAL=()

COMMON_OPTIONS="Options:
  --solution              test the full implementation in src_sol/ and include_sol/
  --src DIR               source tree under test (contains lib/ and app/), default: src/
  --include DIR           include tree under test (contains lib/), default: include/
  --reference LIB         prebuilt reference library (default: tests/reference/prebuilt/libdnn_ref.a
                          if it exists, otherwise the reference is built from src_sol/)
  --reference-from-source build the reference from src_sol/ even if a prebuilt one exists
  --build-dir DIR         build directory, default: build-tests/
  -j N                    parallel build jobs, default: $JOBS
  -v, --verbose           print the full output of failing tests
  -h, --help              show this help"

absolute_path() {
    (cd "$1" 2>/dev/null && pwd) || { echo "error: directory not found: $1" >&2; exit 2; }
}

# parse_common_args <usage text> "$@": sets the variables above; other arguments end up in POSITIONAL.
parse_common_args() {
    local usage="$1"
    shift
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --solution) SRC_DIR="$SOLUTION_SRC_DIR"; INCLUDE_DIR="$SOLUTION_INCLUDE_DIR" ;;
            --src) SRC_DIR="$(absolute_path "$2")"; shift ;;
            --include) INCLUDE_DIR="$(absolute_path "$2")"; shift ;;
            --reference) REFERENCE="$(absolute_path "$(dirname "$2")")/$(basename "$2")"; shift ;;
            --reference-from-source) REFERENCE_FROM_SOURCE=1 ;;
            --build-dir) mkdir -p "$2"; BUILD_DIR="$(absolute_path "$2")"; shift ;;
            -j) JOBS="$2"; shift ;;
            -j*) JOBS="${1#-j}" ;;
            -v|--verbose) VERBOSE=1 ;;
            -h|--help) echo "$usage"; echo; echo "$COMMON_OPTIONS"; exit 0 ;;
            -*) echo "error: unknown option $1" >&2; exit 2 ;;
            *) POSITIONAL+=("$1") ;;
        esac
        shift
    done

    if [[ -z "$REFERENCE" && $REFERENCE_FROM_SOURCE -eq 0 && -f "$PREBUILT_REFERENCE" ]]; then
        REFERENCE="$PREBUILT_REFERENCE"
    fi
    if [[ -n "$REFERENCE" && ! -f "$REFERENCE" ]]; then
        echo "error: reference library not found: $REFERENCE" >&2
        exit 2
    fi
}

# configure [cmake args...]: (re)configures BUILD_DIR for the selected source tree; output only on error.
configure() {
    mkdir -p "$BUILD_DIR"
    if ! cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
            -DDNN_TEST_SRC_DIR="$SRC_DIR" -DDNN_TEST_INCLUDE_DIR="$INCLUDE_DIR" "$@" \
            > "$BUILD_DIR/configure.log" 2>&1; then
        cat "$BUILD_DIR/configure.log" >&2
        echo "error: CMake configuration failed" >&2
        exit 2
    fi
}

# Native build-tool flag to continue past targets that fail to compile.
keep_going_flag() {
    if grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja' "$BUILD_DIR/CMakeCache.txt" 2>/dev/null; then
        echo "-k 0"
    else
        echo "-k"
    fi
}

print_setup() {
    echo "Implementation under test: $SRC_DIR"
    echo "                           $INCLUDE_DIR"
    echo "Build directory:           $BUILD_DIR"
}
