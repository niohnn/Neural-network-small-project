# Unit tests: each test compiles only the files of the implementation under test it needs and
# compares their behaviour with the reference implementation (libdnn_ref).
cmake_minimum_required(VERSION 3.16)
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(DNN_TEST_COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")

add_subdirectory(reference)

# Weak fallbacks for the functions to be implemented (see common/fallbacks/Fallback.h). A static library,
# so an object is only linked if the implementation under test misses some of its functions.
add_library(dnn_test_fallbacks STATIC EXCLUDE_FROM_ALL
        common/fallbacks/MathUtilsFallbacks.cpp
        common/fallbacks/LayerFallbacks.cpp
        common/fallbacks/NetworkFallbacks.cpp
        common/fallbacks/BinaryCrossEntropyFallbacks.cpp)
target_include_directories(dnn_test_fallbacks PRIVATE "${DNN_TEST_INCLUDE_DIR}/lib" "${DNN_TEST_COMMON_DIR}")

# Builds all unit test executables: `cmake --build build --target unit_tests`.
add_custom_target(unit_tests)

# dnn_add_unit_test(<name> [FALLBACKS] SOURCES <files of the implementation under test, relative to lib/>)
# FALLBACKS: link the fallbacks, so that functions the test does not exercise may still be missing.
function(dnn_add_unit_test name)
    cmake_parse_arguments(ARG "FALLBACKS" "" "SOURCES" ${ARGN})
    list(TRANSFORM ARG_SOURCES PREPEND "${DNN_TEST_SRC_DIR}/lib/")

    add_executable(${name} EXCLUDE_FROM_ALL unit/${name}.cpp ${ARG_SOURCES})
    target_include_directories(${name} PRIVATE "${DNN_TEST_INCLUDE_DIR}/lib" "${DNN_TEST_COMMON_DIR}")
    target_link_libraries(${name} PRIVATE dnn_ref GTest::gtest_main OpenMP::OpenMP_CXX)
    if(ARG_FALLBACKS)
        target_link_libraries(${name} PRIVATE dnn_test_fallbacks)
    endif()
    add_dependencies(unit_tests ${name})

    add_test(NAME ${name}
             COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET=${name} -DCONFIG=$<CONFIG>
                     -DEXECUTABLE=$<TARGET_FILE:${name}> -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildAndRun.cmake)
    # the tests build their executables, so they must not run concurrently
    set_tests_properties(${name} PROPERTIES LABELS unit RESOURCE_LOCK dnn_build)
endfunction()

dnn_add_unit_test(t01_activation_headers)
dnn_add_unit_test(t02_activation_functions              SOURCES activation/Sigmoid.cpp activation/ScaledSigmoid.cpp
                                                                activation/ReLU.cpp activation/LeakyReLU.cpp)
dnn_add_unit_test(t03_dot_product_sum         FALLBACKS SOURCES MathUtils.cpp)
dnn_add_unit_test(t04_normalize)
dnn_add_unit_test(t05_layer_forward           FALLBACKS SOURCES Layer.cpp MathUtils.cpp)
dnn_add_unit_test(t06_layer_backward          FALLBACKS SOURCES Layer.cpp MathUtils.cpp)
dnn_add_unit_test(t07_layer_update_parameters FALLBACKS SOURCES Layer.cpp MathUtils.cpp)
dnn_add_unit_test(t08_layer_commit            FALLBACKS SOURCES Layer.cpp MathUtils.cpp)
dnn_add_unit_test(t09_network_construction    FALLBACKS SOURCES Network.cpp Layer.cpp MathUtils.cpp)
dnn_add_unit_test(t10_network_add_layer       FALLBACKS SOURCES Network.cpp Layer.cpp MathUtils.cpp)
dnn_add_unit_test(t11_network_propagation     FALLBACKS SOURCES Network.cpp Layer.cpp MathUtils.cpp)
dnn_add_unit_test(t12_network_training        FALLBACKS SOURCES Network.cpp Layer.cpp MathUtils.cpp)
dnn_add_unit_test(t13_bce_cost                FALLBACKS SOURCES loss/BinaryCrossEntropy.cpp MathUtils.cpp)
dnn_add_unit_test(t14_bce_accuracy_gradient   FALLBACKS SOURCES loss/BinaryCrossEntropy.cpp MathUtils.cpp)
