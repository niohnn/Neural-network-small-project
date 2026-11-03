# Reference implementation for the unit tests: the full implementation compiled with namespace dnn
# renamed to dnn_ref (with internal linkage, see RefUnity.cpp), plus the bridge in RefBridge.h. Either built from source or imported as a
# prebuilt static library (-DDNN_REF_PREBUILT=/path/to/libdnn_ref.a).

set(DNN_REF_PREBUILT "" CACHE FILEPATH "Prebuilt reference library (libdnn_ref.a); empty: build it from DNN_REF_SRC_DIR")
set(DNN_REF_SRC_DIR "${PROJECT_SOURCE_DIR}/src_sol" CACHE PATH "Source directory of the reference implementation")
set(DNN_REF_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include_sol" CACHE PATH "Include directory of the reference implementation")

if(DNN_REF_PREBUILT)
    add_library(dnn_ref STATIC IMPORTED GLOBAL)
    set_target_properties(dnn_ref PROPERTIES IMPORTED_LOCATION "${DNN_REF_PREBUILT}")
    target_include_directories(dnn_ref INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}" "${DNN_TEST_COMMON_DIR}")
    message(STATUS "Unit tests: using prebuilt reference ${DNN_REF_PREBUILT}")
else()
    if(DNN_REF_SRC_DIR STREQUAL DNN_TEST_SRC_DIR)
        message(STATUS "Unit tests: reference and implementation under test are both built from ${DNN_REF_SRC_DIR} "
                       "(self-test). Set DNN_REF_PREBUILT to test against the prebuilt reference.")
    endif()
    # One translation unit (RefUnity.cpp) that includes the reference sources and gives them internal
    # linkage. Optimized, without debug information and stripped (POST_BUILD), so the library only
    # contains the names of the bridge functions in RefBridge.h.
    add_library(dnn_ref STATIC RefUnity.cpp)
    # RefBridge.cpp is included by RefUnity.cpp
    set_source_files_properties(RefBridge.cpp PROPERTIES HEADER_FILE_ONLY ON)
    # renames namespace dnn to dnn_ref, so the reference can be linked next to the implementation under test
    target_compile_definitions(dnn_ref PRIVATE dnn=dnn_ref NDEBUG)
    # -g0 comes after the flags of the build type, so it also overrides -g of a Debug build; -fno-rtti
    # removes the type names of the polymorphic classes (exception types keep theirs)
    target_compile_options(dnn_ref PRIVATE -O2 -g0 -fno-rtti)
    set_target_properties(dnn_ref PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(dnn_ref
            PRIVATE "${DNN_REF_INCLUDE_DIR}/lib" "${DNN_REF_SRC_DIR}"
            PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${DNN_TEST_COMMON_DIR}")
    if(CMAKE_STRIP)
        add_custom_command(TARGET dnn_ref POST_BUILD
                COMMAND "${CMAKE_STRIP}" --strip-unneeded "$<TARGET_FILE:dnn_ref>"
                COMMENT "Stripping the reference library")
    else()
        message(WARNING "Unit tests: no strip tool found, the reference library is not stripped")
    endif()
    target_link_libraries(dnn_ref PRIVATE OpenMP::OpenMP_CXX)
endif()
