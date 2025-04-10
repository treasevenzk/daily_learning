cmake_minimum_required(VERSION 3.10)
project(PackedFun VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include directories
include_directories(include)

# Define source files
set(SOURCES
    src/packed_fun.cpp
    src/registry.cpp
)

# Create shared library
add_library(packedfun SHARED ${SOURCES})

# C API for Python FFI
set(C_API_SOURCES
    src/c_api/packed_fun_api.cpp
    examples/test_functions/test_functions.cpp
)

add_library(packedfun_c_api SHARED ${C_API_SOURCES})
target_link_libraries(packedfun_c_api PRIVATE packedfun)

# Install targets
install(TARGETS packedfun packedfun_c_api
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(DIRECTORY include/
        DESTINATION include/packedfun
        FILES_MATCHING PATTERN "*.h")

# Examples
add_subdirectory(examples)

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()