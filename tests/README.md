# Tests

Two independent kinds of tests:

| Test | What it checks | Needs |
|---|---|---|
| `test_compilation` | All classes/functions exist in `namespace dnn` with the predefined signatures, are defined (link step), and `DNN_app` builds. | compiler only |
| `t01` … `t14` | Functional equivalence of each removed function with the reference implementation. | googletest (fetched at configure time), reference library |

Each unit test compiles only the files it needs, so a test fails on its own while other parts of the
implementation are unfinished (or don't compile yet). Functions of `MathUtils`, `Layer`, `Network` and
`BinaryCrossEntropy` that are not implemented yet are replaced by fallbacks
(`common/fallbacks/`), so the tests still link. Calling a missing function fails with e.g.
`dnn::Layer::updateParameters is not implemented (unit test 7)`.

| # | Test | Functions | Also needs |
|---|---|---|---|
| 1 | `t01_activation_headers` | class declarations in `Sigmoid.h`, `ScaledSigmoid.h`, `ReLU.h`, `LeakyReLU.h` (compile-only) | |
| 2 | `t02_activation_functions` | `Sigmoid.cpp`, `ScaledSigmoid.cpp`, `ReLU.cpp`, `LeakyReLU.cpp` | 1 |
| 3 | `t03_dot_product_sum` | `dotProduct`, `sum` | |
| 4 | `t04_normalize` | both `normalize` templates | |
| 5 | `t05_layer_forward` | `Layer::forwardPropagation` | 3 (if used) |
| 6 | `t06_layer_backward` | `Layer::backwardPropagation` | 3 (if used) |
| 7 | `t07_layer_update_parameters` | `Layer::updateParameters` | |
| 8 | `t08_layer_commit` | `Layer::commitToWeights` | 7 |
| 9 | `t09_network_construction` | `Network` constructor, `~Network` | |
| 10 | `t10_network_add_layer` | `addLayer`, `getLayerInput` | |
| 11 | `t11_network_propagation` | `fullForwardPropagation`, `fullBackwardPropagation` | 5, 6, `getLayerInput` (10) |
| 12 | `t12_network_training` | `trainBatch`, `computeAccuracy` | 5–8, 11, `getLayerInput` (10) |
| 13 | `t13_bce_cost` | `BinaryCrossEntropy::getCostValue` | 3 (if used) |
| 14 | `t14_bce_accuracy_gradient` | `BinaryCrossEntropy::getAccuracyValue`, `getGradient` | |

"Also needs" lists the functions that the tested functions call (in the reference implementation).
The tests otherwise don't depend on each other: the state that other functions would produce is set
directly (e.g. the accumulated gradients for test 8, the forward pass for tests 6 and 10). The Layer
and Network tests use their own activation and loss functions instead of the ones from tests 2, 13 and 14.

On the cluster, first run `module load gcc/13 cmake/4.2`.

## Scripts

The scripts in `tests/scripts/` wrap the CMake/ctest commands below (build directory: `build-tests/`):

```sh
tests/scripts/test_compilation.sh               # compilation / interface check
tests/scripts/run_unit_tests.sh                 # all 14 unit tests
tests/scripts/run_unit_tests.sh 1 2 5           # selected tests, by number ...
tests/scripts/run_unit_tests.sh -v network      # ... or by (part of the) name; -v: show failure details
tests/scripts/run_all_tests.sh                  # both, with a summary
tests/scripts/run_all_tests.sh --solution       # against the full implementation in src_sol/ and include_sol/
tests/scripts/run_unit_tests.sh --src my/src --include my/include
tests/scripts/build_reference.sh                # build tests/reference/prebuilt/libdnn_ref.a from src_sol/
```

If `tests/reference/prebuilt/libdnn_ref.a` exists, the unit tests use it as the reference; otherwise
the reference is built from `src_sol/`. Override with `--reference LIB` or `--reference-from-source`.
All options: `--help`.

## Compilation / interface check

```sh
cmake -S . -B build
cmake --build build --target test_compilation   # compiler errors name the missing/mismatching function
# or via ctest:
ctest --test-dir build -L interface --output-on-failure
```

## Unit tests

```sh
cmake -S . -B build -DDNN_BUILD_TESTS=ON
ctest --test-dir build -L unit                                   # all 14 unit tests
ctest --test-dir build -R t05_layer_forward --output-on-failure  # one test, with details
ctest --test-dir build                                           # unit tests + test_compilation
```

The tests build their own executables, so no separate build step is needed. To build them and run one
directly (prints which sub-case and which value differs from the reference):

```sh
cmake --build build --target unit_tests -- -k     # -k: keep going past tests that don't compile yet
./build/tests/t12_network_training
```

## Testing another source tree

By default `src/` and `include/` (the students' version) are tested. To test e.g. the full
implementation:

```sh
cmake -S . -B build-solution -DDNN_BUILD_TESTS=ON \
      -DDNN_TEST_SRC_DIR=$PWD/src_sol -DDNN_TEST_INCLUDE_DIR=$PWD/include_sol
ctest --test-dir build-solution
```

## Prebuilt reference library

Without further options, the reference is built from `src_sol/` and `include_sol/` (`DNN_REF_SRC_DIR`,
`DNN_REF_INCLUDE_DIR`). If `src_sol/` is the implementation under test, this is only a self-test (a configure message says so). To ship the
reference without its source, build it once with the same compiler as the students (gcc 13), then pass
it via `DNN_REF_PREBUILT`. The scripts do this automatically if `tests/reference/prebuilt/libdnn_ref.a` exists:

```sh
tests/scripts/build_reference.sh        # with the full implementation -> tests/reference/prebuilt/libdnn_ref.a

# with the student implementation:
cmake -S . -B build -DDNN_BUILD_TESTS=ON -DDNN_REF_PREBUILT=$PWD/tests/reference/prebuilt/libdnn_ref.a
ctest --test-dir build
```

The reference is the full implementation compiled as one translation unit (`reference/RefUnity.cpp`)
with `namespace dnn` renamed to `dnn_ref` and wrapped in an anonymous namespace. It is compiled with
`-O2 -g0 -fno-rtti -DNDEBUG` and stripped (`strip --strip-unneeded`). Only the bridge functions in
`namespace dnnref` (`reference/RefBridge.h`) keep their names; `build_reference.sh` checks this and
that there is no debug information. Stripping removes names and debug information and makes
decompilation harder, but machine code can still be disassembled. The library also contains the
string literals of the implementation, e.g. the activation names.

Rebuild the prebuilt library whenever `tests/common/` or `tests/reference/` changes, since it contains
the compiled test harness.

### Using the prebuilt library on other systems

The library is position-independent and uses only baseline x86-64 instructions (no `-march`). It can
be used on any system with:

- Linux on x86-64 (any distribution, the cluster, WSL);
- g++ 13 or newer, or clang with libstdc++ of gcc ≥ 13 (the bridge passes `std::vector`/`std::string`;
  an older g++ may fail to link with "undefined reference");
- the `tests/common/` and `tests/reference/` it was built with.

It does not work on macOS, Windows (MSVC/MinGW), ARM (e.g. Apple Silicon) or with clang + libc++.
Build it with the oldest compiler it will be used with, i.e. with `gcc/13` on the cluster. To check a
system, run the tests against the full implementation; they must pass 14/14:

```sh
tests/scripts/run_unit_tests.sh --solution --reference path/to/libdnn_ref.a
```
