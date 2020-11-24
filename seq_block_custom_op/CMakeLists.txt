cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(plugins LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)

# Define our library target
add_library(plugins SHARED bindings.cpp)

# Enable C++14
target_compile_features(plugins PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(plugins "${TORCH_LIBRARIES}")

set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
