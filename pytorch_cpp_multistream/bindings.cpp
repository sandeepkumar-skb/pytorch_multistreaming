#include <torch/extension.h>
#include "run_model_extension.cpp"
#include "run_multi_model_extension.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &d_launch, "wrapper for d_launch");
    m.def("multi_launch", &d_multi_launch, "wrapper for d_launch");
}
