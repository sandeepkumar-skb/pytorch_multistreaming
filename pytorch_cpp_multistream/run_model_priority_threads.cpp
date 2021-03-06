#include <torch/script.h>
#include <sstream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <thread>
#include <chrono>
#include <pthread.h>
//#include <torch/torch.h>
//#include <cuda_runtime_api.h>
#include <memory>
#include <iostream>
#include <vector>
#include <future>

class thread : public std::thread
{
  public:
    static void setScheduling(std::thread &th, int priority) {
        sched_param sch;
        sch.sched_priority = priority;
        if(pthread_setschedparam(th.native_handle(), SCHED_RR, &sch)) {
            std::cerr << "Failed to set Thread scheduling : " << std::strerror(errno) << std::endl;
        }
    }

};

void launch_kernel(at::cuda::CUDAStream& torch_stream,  torch::jit::script::Module& module, std::vector<torch::jit::IValue>& inputs){
    at::cuda::CUDAStreamGuard torch_guard(torch_stream);
    module.forward(inputs).toTensor();
    //sched_param sch;
    //int policy;
    //pthread_getschedparam(pthread_self(), &policy, &sch);
    //std::cout << "Thread priorit is set to: " << sch.sched_priority << '\n';
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
    std::thread thread1;
    std::thread thread2;

    //WARM UP
    for(int i=0; i<50; ++i){
        thread1 = std::thread(launch_kernel, std::ref(torch_stream1), std::ref(model1), std::ref(inputs));
        thread2 = std::thread(launch_kernel, std::ref(torch_stream2), std::ref(model2), std::ref(inputs));
        thread1.join();
        thread2.join();
    }
    cudaDeviceSynchronize();
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end ;
    std::chrono::duration<double> span;
 
    int num_iter = 1000;
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<num_iter; ++i){

        thread1 = std::thread(launch_kernel, std::ref(torch_stream1), std::ref(model1), std::ref(inputs));
        thread::setScheduling(thread1, 99);
        //thread2 = std::thread(launch_kernel, std::ref(torch_stream2), std::ref(model2), std::ref(inputs));
        //thread::setScheduling(thread2, 99);
        model2.forward(inputs);
        
        thread1.join();
        //thread2.join();

        /*
        std::vector<std::thread> v;
        v.emplace_back(launch_kernel, std::ref(torch_stream1), std::ref(model1), std::ref(inputs));
        v.emplace_back(launch_kernel, std::ref(torch_stream2), std::ref(model2), std::ref(inputs));
        for (auto& t : v){
            t.join();
        }
        */
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time per Iteration: " << (span.count()*1000)/num_iter << "ms" << std::endl;

    std::cout << "ok\n";
}

