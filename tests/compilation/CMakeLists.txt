# test_compilation: checks that the implementation under test compiles, declares every function
# with the predefined signature (InterfaceCheck.cpp), defines all of them (link step), and that the
# application builds. Independent of the unit tests and of googletest.
#
#   cmake --build build --target test_compilation     or     ctest --test-dir build -L interface

set(dnn_test_lib_sources
        "${DNN_TEST_SRC_DIR}/lib/activation/Sigmoid.cpp"
        "${DNN_TEST_SRC_DIR}/lib/activation/ScaledSigmoid.cpp"
        "${DNN_TEST_SRC_DIR}/lib/activation/ReLU.cpp"
        "${DNN_TEST_SRC_DIR}/lib/activation/LeakyReLU.cpp"
        "${DNN_TEST_SRC_DIR}/lib/loss/BinaryCrossEntropy.cpp"
        "${DNN_TEST_SRC_DIR}/lib/Layer.cpp"
        "${DNN_TEST_SRC_DIR}/lib/Network.cpp"
        "${DNN_TEST_SRC_DIR}/lib/MathUtils.cpp"
        "${DNN_TEST_SRC_DIR}/lib/IOUtils.cpp"
)

add_library(test_compilation_lib OBJECT EXCLUDE_FROM_ALL ${dnn_test_lib_sources})
target_include_directories(test_compilation_lib PUBLIC "${DNN_TEST_INCLUDE_DIR}/lib")
target_link_libraries(test_compilation_lib PUBLIC OpenMP::OpenMP_CXX)

# separate object library, so the signature checks are compiled (and reported) even if the library fails
add_library(test_compilation_interface OBJECT EXCLUDE_FROM_ALL InterfaceCheck.cpp)
target_include_directories(test_compilation_interface PRIVATE "${DNN_TEST_INCLUDE_DIR}/lib")

add_executable(test_compilation_check EXCLUDE_FROM_ALL)
target_link_libraries(test_compilation_check PRIVATE test_compilation_interface test_compilation_lib)

add_executable(test_compilation_app EXCLUDE_FROM_ALL "${DNN_TEST_SRC_DIR}/app/main.cpp")
target_link_libraries(test_compilation_app PRIVATE test_compilation_lib)

add_custom_target(test_compilation
        COMMAND test_compilation_check
        COMMENT "Checking interface and compilation of ${DNN_TEST_SRC_DIR}")
add_dependencies(test_compilation test_compilation_check test_compilation_app)

add_test(NAME test_compilation
         COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${CMAKE_BINARY_DIR} -DTARGET=test_compilation -DCONFIG=$<CONFIG>
                 -P ${PROJECT_SOURCE_DIR}/tests/cmake/BuildAndRun.cmake)
set_tests_properties(test_compilation PROPERTIES LABELS interface RESOURCE_LOCK dnn_build)
