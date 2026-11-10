# CMake generated Testfile for 
# Source directory: /mnt/c/Users/Anwender/Desktop/school/pp/pp-P1-main/tests/compilation
# Build directory: /mnt/c/Users/Anwender/Desktop/school/pp/pp-P1-main/build-wsl/tests/compilation
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_compilation "/usr/bin/cmake" "-DBUILD_DIR=/mnt/c/Users/Anwender/Desktop/school/pp/pp-P1-main/build-wsl" "-DTARGET=test_compilation" "-DCONFIG=" "-P" "/mnt/c/Users/Anwender/Desktop/school/pp/pp-P1-main/tests/cmake/BuildAndRun.cmake")
set_tests_properties(test_compilation PROPERTIES  LABELS "interface" RESOURCE_LOCK "dnn_build" _BACKTRACE_TRIPLES "/mnt/c/Users/Anwender/Desktop/school/pp/pp-P1-main/tests/compilation/CMakeLists.txt;38;add_test;/mnt/c/Users/Anwender/Desktop/school/pp/pp-P1-main/tests/compilation/CMakeLists.txt;0;")
