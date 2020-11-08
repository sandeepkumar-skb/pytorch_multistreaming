#include <torch/script.h>
#include <sstream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <thread>
#include <chrono>
#include <pthread.h>
#include <tbb/tbb.h>

void launch_kernel(at::cuda::CUDAStream& torch_stream,  torch::jit::script::Module& module, std::vector<torch::jit::IValue>& inputs){
    at::cuda::CUDAStreamGuard torch_guard(torch_stream);
    module.forward(inputs).toTensor();
}


int main(){
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
        return -1;
    }
 
    at::TensorOptions tensor_options;
    tensor_options = tensor_options.dtype(c10::kFloat);
    tensor_options = tensor_options.device(c10::kCUDA);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({64, 64}, tensor_options));
    //inputs.push_back(torch::ones({1, 3, 224, 224}, tensor_options));
    model1.to(at::kCUDA);
    model2.to(at::kCUDA);
    at::cuda::CUDAStream torch_stream1 = at::cuda::getStreamFromPool();
    at::cuda::CUDAStream torch_stream2 = at::cuda::getStreamFromPool();
    tbb::task_scheduler_init init(2);
    tbb::task_group group;

    //WARM UP
    for(int i=0; i<5; ++i){
        group.run([&](){
                at::cuda::CUDAStreamGuard torch_guard(torch_stream1);
                model1.forward(inputs);
                });
        group.run([&](){
            at::cuda::CUDAStreamGuard torch_guard(torch_stream2);
            model2.forward(inputs);
            });
    }
    group.wait();
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end ;
    std::chrono::duration<double> span;
 
    int num_iter = 500;
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<num_iter; ++i){
        group.run([&](){
            at::cuda::CUDAStreamGuard torch_guard(torch_stream1);
            model1.forward(inputs);
            //at::cuda::CUDAStreamGuard torch_guard2(torch_stream2);
            //model2.forward(inputs);
            });
        group.run([&](){
            at::cuda::CUDAStreamGuard torch_guard(torch_stream2);
            model2.forward(inputs);
            });
    }
    group.wait();

    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time per Iteration: " << (span.count()*1000)/num_iter << "ms" << std::endl;

    std::cout << "ok\n";
}

