cmake_minimum_required(VERSION 3.18)
project(fluid_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Fetch pybind11 from GitHub
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# To offset LTO incompatibilities with CUDA: 
# NOTE: enable LTO for production!
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

# Build the core simulation library
add_library(fluid_sim STATIC
    src/simulation.cpp
    src/kernels/simulation.cu
)

set_source_files_properties(src/kernels/simulation.cu PROPERTIES LANGUAGE CUDA)

target_include_directories(fluid_sim
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src/kernels
)

set_target_properties(fluid_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
)

# Build the pybind11 module
pybind11_add_module(pyfluid src/bindings.cpp)
target_link_libraries(pyfluid PRIVATE fluid_sim)
