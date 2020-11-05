cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

#add_executable(run_model run_model_async_launch.cpp)
add_executable(run_model run_model_threads.cpp)
#add_executable(run_model run_model.cpp)
#add_executable(run_model run_multi_model.cpp)
target_link_libraries(run_model "${TORCH_LIBRARIES}")
set_property(TARGET run_model PROPERTY CXX_STANDARD 14)
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")
