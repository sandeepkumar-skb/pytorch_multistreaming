#include <torch/script.h>
#include <sstream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <chrono>
#include <memory>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <future>
#include <torch/custom_class.h>

struct MyLaunchClass : torch::CustomClassHolder {
    torch::jit::script::Module model;
    std::vector<torch::jit::script::Module>  sub_model;
    std::string path;
    MyLaunchClass(std::string path) : path{path} {
        if (path.size() == 0) return ;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            std::cout << "Trying to load the model from path: " << path << std::endl;
            model = torch::jit::load(path);
            model.eval();
            model.to(at::kCUDA);
        }
        catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
        }

        std::unordered_set<std::string> module_names {"block1", "block2"};
        for (auto it : model.named_modules()){
            if (module_names.count(it.name) != 0){
                sub_model.push_back(it.value);
            }
        }

    }

    std::vector<at::Tensor> d_multi_launch(torch::Tensor inp1, torch::Tensor inp2)
    {
        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end ;
        std::chrono::duration<double> span;
        start = std::chrono::high_resolution_clock::now();
        at::TensorOptions tensor_options;
        tensor_options = tensor_options.dtype(c10::kFloat);
        tensor_options = tensor_options.device(c10::kCUDA);
        std::vector<std::vector<torch::jit::IValue>> inputs;
        inputs.push_back(std::vector<torch::jit::IValue> {inp1});
        inputs.push_back(std::vector<torch::jit::IValue> {inp2});
        std::vector<at::Tensor> outputs;
    
        std::vector<at::cuda::CUDAStream> cuda_streams;
        for (int i=0; i < sub_model.size(); ++i){
            cuda_streams.push_back(std::move(at::cuda::getStreamFromPool()));
        }
    
        for (int model_id=0; model_id< sub_model.size(); ++model_id){
            at::cuda::CUDAStreamGuard guard(cuda_streams[model_id]);
            outputs.push_back(sub_model[model_id].forward(inputs[model_id]).toTensor());
        }
        cudaDeviceSynchronize();
        end = std::chrono::high_resolution_clock::now();
        span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "CPP total time: " << span.count()*1000 << "ms" << std::endl;
    
        return outputs;
    }
};

