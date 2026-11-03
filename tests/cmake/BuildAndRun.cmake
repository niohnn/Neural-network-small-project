# Builds TARGET in BUILD_DIR and runs EXECUTABLE (if given). Used as a ctest command, so that a
# test whose implementation does not compile yet fails on its own instead of breaking the build.
#
#   cmake -DBUILD_DIR=<dir> -DTARGET=<target> [-DCONFIG=<config>] [-DEXECUTABLE=<path>] -P BuildAndRun.cmake

set(build_command "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target "${TARGET}")
if(CONFIG)
    list(APPEND build_command --config "${CONFIG}")
endif()

execute_process(COMMAND ${build_command} RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Building ${TARGET} failed (see compiler output above).")
endif()

if(EXECUTABLE)
    execute_process(COMMAND "${EXECUTABLE}" RESULT_VARIABLE run_result)
    if(NOT run_result EQUAL 0)
        message(FATAL_ERROR "${TARGET} failed.")
    endif()
endif()
