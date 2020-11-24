#include <torch/torch.h>
#include <torch/script.h>
#include "run_multi_model_extension.cpp"

TORCH_LIBRARY (plugins, m){
    m.def("multi_launch", d_multi_launch); 
}
