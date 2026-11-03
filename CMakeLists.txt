cmake_minimum_required(VERSION 3.16)
project(DNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)


set(dnn_libsrc
        src/lib/activation/Sigmoid.cpp
        src/lib/activation/ScaledSigmoid.cpp
        src/lib/activation/ReLU.cpp
        src/lib/activation/LeakyReLU.cpp
        src/lib/loss/BinaryCrossEntropy.cpp
        src/lib/Layer.cpp
        src/lib/Network.cpp
        src/lib/MathUtils.cpp
        src/lib/IOUtils.cpp
)


add_library(dnn_objlib OBJECT ${dnn_libsrc})
set_target_properties(dnn_objlib PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(dnn_objlib PUBLIC include/lib)

add_library(DNN_so SHARED)
set_target_properties(DNN_so PROPERTIES OUTPUT_NAME "DNN")
target_link_libraries(DNN_so PUBLIC dnn_objlib)

add_library(DNN_a STATIC)
set_target_properties(DNN_a PROPERTIES OUTPUT_NAME "DNN")
target_link_libraries(DNN_a PUBLIC dnn_objlib)

add_executable(DNN_app src/app/main.cpp)
target_link_libraries(DNN_app PRIVATE DNN_so)

# ---- Tests ----
# Implementation under test (directories containing lib/ and app/ resp. lib/).
set(DNN_TEST_SRC_DIR "${PROJECT_SOURCE_DIR}/src" CACHE PATH "Source directory of the implementation under test")
set(DNN_TEST_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include" CACHE PATH "Include directory of the implementation under test")

enable_testing()

# Compilation / interface check: `cmake --build build --target test_compilation` or `ctest -L interface`.
add_subdirectory(tests/compilation)

# Unit tests against the reference implementation: configure with -DDNN_BUILD_TESTS=ON.
option(DNN_BUILD_TESTS "Build the unit tests (fetches googletest)" OFF)
if(DNN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
