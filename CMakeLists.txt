cmake_minimum_required(VERSION 3.16)
project(lmmse_demosaic LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build with optimizations by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Find OpenMP
find_package(OpenMP REQUIRED)

# Create executable
add_executable(lmmse_demosaic lmmse_demosaic.cpp)

# Link OpenMP
target_link_libraries(lmmse_demosaic PRIVATE OpenMP::OpenMP_CXX)

# Enable warnings
target_compile_options(lmmse_demosaic PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>
)

# Print build info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "OpenMP Found: ${OpenMP_CXX_FOUND}")
