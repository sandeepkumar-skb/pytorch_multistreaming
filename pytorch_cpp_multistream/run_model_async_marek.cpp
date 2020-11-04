#include <torch/script.h>
#include <sstream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
//#include <torch/torch.h>
//#include <cuda_runtime_api.h>

#include <memory>
#include <iostream>
#include <vector>
#include <future>

int launch(std::shared_ptr<at::cuda::CUDAStream> stream, std::shared_ptr<torch::jit::script::Module> module, std::shared_ptr<std::vector<torch::jit::IValue>> inputs) {
    at::cuda::CUDAStreamGuard torch_guard(*stream);
    module->forward(*inputs);
    return 0;
}

int main(){
    std::shared_ptr<torch::jit::script::Module> model1;
    std::shared_ptr<torch::jit::script::Module> model2;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model1 = std::make_shared<torch::jit::script::Module>(torch::jit::load("/home/sandeep.skb/workspace/misc/pytorch_experiments/cpp_multistreaming/dummy_net.pth"));
        model2 = std::make_shared<torch::jit::script::Module>(torch::jit::load("/home/sandeep.skb/workspace/misc/pytorch_experiments/cpp_multistreaming/dummy_net.pth"));
        model1->eval();
        model2->eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
 
    at::TensorOptions tensor_options;
    tensor_options = tensor_options.dtype(c10::kFloat);
    tensor_options = tensor_options.device(c10::kCUDA);
    std::vector<torch::jit::IValue> in;
    //inputs.push_back(torch::ones({1, 3, 224, 224}, tensor_options));
    in.push_back(torch::ones({512, 512}, tensor_options));
    auto inputs = std::make_shared<std::vector<torch::jit::IValue>>(in);
    model1->to(at::kCUDA);
    model2->to(at::kCUDA);

    std::shared_ptr<at::cuda::CUDAStream> torch_stream1 = std::make_shared<at::cuda::CUDAStream>(at::cuda::getStreamFromPool());
    std::shared_ptr<at::cuda::CUDAStream> torch_stream2 = std::make_shared<at::cuda::CUDAStream>(at::cuda::getStreamFromPool());

    for(int i=0; i<5; ++i){
        std::async(std::launch::async, launch, torch_stream1, model1, inputs);
        std::async(std::launch::async, launch, torch_stream2, model2, inputs);
    }
    cudaDeviceSynchronize();

    std::cout << "ok\n";
}

