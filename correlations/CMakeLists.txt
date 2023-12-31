cmake_minimum_required(VERSION 3.5)
project(correlations LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# In debug mode, add -DDEBUG
add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    # Warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

    # Remove Asio warning
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")

    # Static library: -fPIC
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -march=native -fstack-protector")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# sum_correlations

add_executable(sum_correlations
    sum_correlations.cpp
    mmapfile.cpp
    mmapfile.hpp
    compress.cpp
    compress.hpp
)
target_link_libraries(sum_correlations PUBLIC zstd)

# study_co_activations

add_executable(study_co_activations
study_co_activations.cpp
    mmapfile.cpp
    mmapfile.hpp
    compress.cpp
    compress.hpp
    co_activations.cpp
    co_activations.hpp
)
target_include_directories(study_co_activations PUBLIC ${MKL_INCLUDE_DIR})
target_link_libraries(study_co_activations PUBLIC zstd ${OpenCV_LIBS} Eigen3::Eigen)
