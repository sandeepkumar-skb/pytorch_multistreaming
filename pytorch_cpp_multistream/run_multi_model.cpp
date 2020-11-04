#include <torch/script.h>
#include <sstream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <chrono>
//#include <torch/torch.h>
//#include <cuda_runtime_api.h>

#include <memory>
#include <iostream>
#include <vector>
#include <future>
/*
int launch(std::shared_ptr<at::cuda::CUDAStream> stream, std::shared_ptr<torch::jit::script::Module> module, std::shared_ptr<std::vector<torch::jit::IValue>> inputs) {
    at::cuda::CUDAStreamGuard torch_guard(*stream);
    module->forward(*inputs);
    return 0;
}
*/

int main(){
    torch::jit::script::Module model;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model = torch::jit::load("./seq_multi_block.pth");
        model.eval();
        model.to(at::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::vector<torch::jit::script::Module>  sub_model;
    std::unordered_set<std::string> module_names {"block1", "block2"};
    for (auto it : model.named_modules()){
        if (module_names.count(it.name) != 0){
            std::cout << "Found " << it.name << std::endl;
            sub_model.push_back(it.value);
        }
    }
    at::TensorOptions tensor_options;
    tensor_options = tensor_options.dtype(c10::kFloat);
    tensor_options = tensor_options.device(c10::kCUDA);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({512, 512}, tensor_options));

    std::vector<at::cuda::CUDAStream> cuda_streams;
    for (int i=0; i < sub_model.size(); ++i){
        cuda_streams.push_back(std::move(at::cuda::getStreamFromPool()));
    }

    // Warm UP
    for(int iter=0; iter<50; ++iter){
        for (int model_id=0; model_id< sub_model.size(); ++model_id){
            at::cuda::CUDAStreamGuard guard(cuda_streams[model_id]);
            sub_model[model_id].forward(inputs).toTensor();
        }
    }
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end ;
    std::chrono::duration<double> span;
    start = std::chrono::high_resolution_clock::now();
    for(int iter=0; iter<50; ++iter){
        for (int model_id=0; model_id< sub_model.size(); ++model_id){
            at::cuda::CUDAStreamGuard guard(cuda_streams[model_id]);
            sub_model[model_id].forward(inputs).toTensor();
        }
    }
    cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();
    span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Total time: " << span.count()*1000 << "ms" << std::endl;
    std::cout << "ok\n";
}

