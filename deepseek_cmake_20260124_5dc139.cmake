%%writefile /content/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(vedic_engine VERSION 5.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable all warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wpedantic")

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")

# Find packages
find_package(Boost 1.70 REQUIRED COMPONENTS 
    serialization 
    multiprecision 
    system 
    filesystem 
    thread 
    program_options 
    unit_test_framework
)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA 11.0 QUIET)
find_package(pybind11 2.6 QUIET)
find_package(GTest QUIET)

# Configuration options
option(VEDIC_ENABLE_CUDA "Enable CUDA backend" ${CUDA_FOUND})
option(VEDIC_ENABLE_PYTHON "Enable Python bindings" ${pybind11_FOUND})
option(VEDIC_ENABLE_TESTS "Enable tests" ${GTest_FOUND})
option(VEDIC_ENABLE_OPENMP "Enable OpenMP" ${OpenMP_FOUND})

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(VEDIC_ENABLE_OPENMP)
    include_directories(${OpenMP_CXX_INCLUDE_DIRS})
endif()

if(VEDIC_ENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    find_package(CUDNN QUIET)
    find_package(CUBLAS QUIET)
endif()

if(VEDIC_ENABLE_PYTHON)
    include_directories(${pybind11_INCLUDE_DIRS})
endif()

# Source files
set(VEDIC_SOURCES
    src/core/vedic_engine.cpp
    src/core/fixed256.cpp
    src/core/rational.cpp
    src/core/sutras.cpp
    src/fabric/kronecker_fabric.cpp
    src/fabric/hypercube_lattice.cpp
    src/operators/omega_operator.cpp
    src/fields/theta_field.cpp
    src/optimization/bayesian_optimizer.cpp
    src/visualization/tensor_visualizer.cpp
    src/persistence/state_manager.cpp
    src/parallel/thread_pool.cpp
    src/parallel/openmp_matrix.cpp
)

if(VEDIC_ENABLE_CUDA)
    set(VEDIC_SOURCES ${VEDIC_SOURCES}
        src/cuda/cuda_fixed256.cu
        src/cuda/cuda_matrix.cu
        src/cuda/cuda_kronecker.cu
        src/cuda/cuda_hypercube.cu
    )
endif()

# Create library
add_library(vedic_engine STATIC ${VEDIC_SOURCES})

target_link_libraries(vedic_engine
    ${Boost_LIBRARIES}
    Eigen3::Eigen
)

if(VEDIC_ENABLE_OPENMP)
    target_link_libraries(vedic_engine OpenMP::OpenMP_CXX)
endif()

if(VEDIC_ENABLE_CUDA)
    target_link_libraries(vedic_engine
        ${CUDA_LIBRARIES}
        ${CUDA_cublas_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${CUDNN_LIBRARY}
    )
    set_property(TARGET vedic_engine PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif()

# Main executable
add_executable(vedic_main src/main.cpp)
target_link_libraries(vedic_main vedic_engine)

# Python bindings
if(VEDIC_ENABLE_PYTHON)
    add_subdirectory(python)
endif()

# Tests
if(VEDIC_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Install
install(TARGETS vedic_engine DESTINATION lib)
install(FILES include/vedic_complete.hpp DESTINATION include/vedic)
install(FILES ${VEDIC_HEADERS} DESTINATION include/vedic)