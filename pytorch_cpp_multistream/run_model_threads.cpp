#include <torch/script.h>
#include <sstream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <thread>
#include <chrono>
//#include <torch/torch.h>
//#include <cuda_runtime_api.h>

#include <memory>
#include <iostream>
#include <vector>
#include <future>

void launch_kernel(at::cuda::CUDAStream& torch_stream,  torch::jit::script::Module& module, std::vector<torch::jit::IValue>& inputs){
    at::cuda::CUDAStreamGuard torch_guard(torch_stream);
    module.forward(inputs).toTensor();
}


int main(){
    torch::jit::script::Module model1;
    torch::jit::script::Module model2;
    torch::jit::script::Module model3;
    torch::jit::script::Module model4;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model1 = torch::jit::load("./seq_block.pth");
        model2 = torch::jit::load("./seq_block.pth");
        model3 = torch::jit::load("./seq_block.pth");
        model4 = torch::jit::load("./seq_block.pth");
        model1.eval();
        model2.eval();
        model3.eval();
        model4.eval();
        model1.to(at::kCUDA);
        model2.to(at::kCUDA);
        model3.to(at::kCUDA);
        model4.to(at::kCUDA);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
 
    at::TensorOptions tensor_options;
    tensor_options = tensor_options.dtype(c10::kFloat);
    tensor_options = tensor_options.device(c10::kCUDA);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({512, 512}, tensor_options));
    //inputs.push_back(torch::ones({1, 3, 224, 224}, tensor_options));
    at::cuda::CUDAStream torch_stream1 = at::cuda::getStreamFromPool();
    at::cuda::CUDAStream torch_stream2 = at::cuda::getStreamFromPool();
    at::cuda::CUDAStream torch_stream3 = at::cuda::getStreamFromPool();
    at::cuda::CUDAStream torch_stream4 = at::cuda::getStreamFromPool();
    std::thread thread1;
    std::thread thread2;
    std::thread thread3;
    std::thread thread4;

    int num_threads = 2; // Thread waves - in each wave there are 4 threads
    //WARM UP
    for(int i=0; i<50; ++i){
        /*
        thread1 = std::thread(launch_kernel, std::ref(torch_stream1), std::ref(model1), std::ref(inputs));
        thread2 = std::thread(launch_kernel, std::ref(torch_stream2), std::ref(model2), std::ref(inputs));
        thread1.join();
        thread2.join();
        */
        std::vector<std::thread> v;
        for (int t_id=0; t_id < num_threads; ++t_id){
            v.emplace_back(launch_kernel, std::ref(torch_stream1), std::ref(model1), std::ref(inputs));
            v.emplace_back(launch_kernel, std::ref(torch_stream2), std::ref(model2), std::ref(inputs));
            v.emplace_back(launch_kernel, std::ref(torch_stream3), std::ref(model3), std::ref(inputs));
            v.emplace_back(launch_kernel, std::ref(torch_stream4), std::ref(model4), std::ref(inputs));
        }
        for (auto& t : v){
            t.join();
        }

    }
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end ;
    std::chrono::duration<double> span;
    start = std::chrono::high_resolution_clock::now();
    int num_iter = 1000/num_threads;
    for(int i=0; i<num_iter; ++i){
        /*
        thread1 = std::thread(launch_kernel, std::ref(torch_stream1), std::ref(model1), std::ref(inputs));
        thread2 = std::thread(launch_kernel, std::ref(torch_stream2), std::ref(model2), std::ref(inputs));
        thread1.join();
        thread2.join();
        */
        std::vector<std::thread> v;
        for (int t_id=0; t_id < num_threads; ++t_id){
            v.emplace_back(launch_kernel, std::ref(torch_stream1), std::ref(model1), std::ref(inputs));
            v.emplace_back(launch_kernel, std::ref(torch_stream2), std::ref(model2), std::ref(inputs));
            v.emplace_back(launch_kernel, std::ref(torch_stream3), std::ref(model3), std::ref(inputs));
            v.emplace_back(launch_kernel, std::ref(torch_stream4), std::ref(model4), std::ref(inputs));
        }
        for (auto& t : v){
            t.join();
        }
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Iter time: " << (span.count()*1000)/num_iter << "ms" << std::endl;

    std::cout << "ok\n";
}

