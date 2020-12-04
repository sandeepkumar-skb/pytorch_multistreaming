cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_class LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(OpenMP REQUIRED)

# Define our library target
add_library(custom_class SHARED bindings.cpp)

# Enable C++14
target_compile_features(custom_class PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(custom_class  "${OpenMP_CXX_FLAGS}"
                                    "${TORCH_LIBRARIES}")
target_compile_options(custom_class PRIVATE "${OpenMP_CXX_FLAGS}")

set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
