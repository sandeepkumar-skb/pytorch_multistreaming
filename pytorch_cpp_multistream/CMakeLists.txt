cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

set(EXTERNAL_INSTALL_DIR ${CMAKE_BINARY_DIR}/external)
link_directories(${EXTERNAL_INSTALL_DIR}/lib)
include_directories(${EXTERNAL_INSTALL_DIR}/include)

#add_executable(run_model run_model_async_launch.cpp)
#add_executable(run_model run_model_threads.cpp)
#add_executable(run_model run_model_priority_threads.cpp)
add_executable(run_model run_model_tbb2_threads.cpp)
#add_executable(run_model run_model.cpp)
#add_executable(run_model run_model_baseline.cpp)
#add_executable(run_model run_multi_model.cpp)
target_link_libraries(run_model tbb
                                "${TORCH_LIBRARIES}")
set_property(TARGET run_model PROPERTY CXX_STANDARD 14)
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")


######################################################################
# Build rules for tbb
######################################################################

include(ExternalProject)
ExternalProject_Add(
	tbb
	GIT_REPOSITORY ${CMAKE_SOURCE_DIR}/external/tbb
	GIT_TAG master
	INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
	CMAKE_ARGS
	-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
	-DTBB_BUILD_SHARED=ON
	-DTBB_BUILD_STATIC=OFF
	-DTBB_BUILD_TBBMALLOC=OFF
	-DTBB_BUILD_TBBMALLOC_PROXY=OFF
	-DTBB_BUILD_TESTS=OFF
	-DTBB_CI_BUILD=OFF
)
add_dependencies(run_model tbb)
