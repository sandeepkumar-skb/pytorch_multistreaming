#include <torch/torch.h>
#include <torch/script.h>
#include "run_multi_model_extension.cpp"

TORCH_LIBRARY (plugin_class, m){
    m.class_<MyLaunchClass>("MyLaunchClass")
        .def(torch::init<std::string>())
        .def("multi_launch", &MyLaunchClass::d_multi_launch)
        .def_pickle(
            // __getstate__
            [](const c10::intrusive_ptr<MyLaunchClass>& self) 
                -> std::string
                {
                return self->path;
                },
            // __setstate__
            [](std::string state)
                -> c10::intrusive_ptr<MyLaunchClass>{
                return c10::make_intrusive<MyLaunchClass>(std::move(state));
                }
            )

    ; 
}

