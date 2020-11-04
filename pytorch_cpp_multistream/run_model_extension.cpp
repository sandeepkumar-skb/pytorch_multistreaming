#include <torch/script.h>
#include <sstream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <chrono>
#include <memory>
#include <iostream>
#include <vector>
#include <utility>

std::pair<at::Tensor, at::Tensor> d_launch(at::Tensor input_tensor){
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end ;
    std::chrono::duration<double> span;
    start = std::chrono::high_resolution_clock::now();
    torch::jit::script::Module model1;
    torch::jit::script::Module model2;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model1 = torch::jit::load("./seq_block.pth");
        model2 = torch::jit::load("./seq_block.pth");
        model1.eval();
        model2.eval();
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
 
    at::TensorOptions tensor_options;
    tensor_options = tensor_options.dtype(c10::kFloat);
    tensor_options = tensor_options.device(c10::kCUDA);
    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(input_tensor.to(tensor_options));
    //inputs.push_back(input_tensor);
    //inputs.push_back(torch::ones({1, 3, 224, 224}, tensor_options));
    //inputs.push_back(torch::ones({512, 512}, tensor_options));
    model1.to(at::kCUDA);
    model2.to(at::kCUDA);

    at::cuda::CUDAStream torch_stream1 = at::cuda::getStreamFromPool();
    at::cuda::CUDAStream torch_stream2 = at::cuda::getStreamFromPool();
    at::Tensor output1;
    at::Tensor output2;

    {
        at::cuda::CUDAStreamGuard torch_guard1(torch_stream1);
        output1 = model1.forward(inputs).toTensor();
    }
    {
        at::cuda::CUDAStreamGuard torch_guard2(torch_stream2);
        output2 = model2.forward(inputs).toTensor();
    }
    cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();
    span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "CPP total time: " << span.count()*1000 << "ms" << std::endl;
    return std::make_pair(output1, output2);
}

