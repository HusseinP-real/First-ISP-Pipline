cmake_minimum_required(VERSION 3.10)
project(BM3D_Denoiser VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find OpenMP (optional, for parallelization)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling multi-threaded processing")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(STATUS "OpenMP not found - building single-threaded version")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Library
add_library(bm3d_lib STATIC
    src/bm3d_denoiser.cpp
)

target_link_libraries(bm3d_lib
    ${OpenCV_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bm3d_lib OpenMP::OpenMP_CXX)
endif()

# Executable
add_executable(bm3d_denoise main.cpp)

target_link_libraries(bm3d_denoise
    bm3d_lib
    ${OpenCV_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(bm3d_denoise OpenMP::OpenMP_CXX)
endif()

# Installation
install(TARGETS bm3d_denoise bm3d_lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES include/bm3d_denoiser.hpp
    DESTINATION include
)

# Print configuration
message(STATUS "")
message(STATUS "BM3D Denoiser Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenCV version: ${OpenCV_VERSION}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "")
