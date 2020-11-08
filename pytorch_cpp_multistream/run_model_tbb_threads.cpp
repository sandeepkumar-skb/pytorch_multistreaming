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
#include <thread>
#include <tbb/tbb.h>
using namespace tbb;


class Launch{
    std::shared_ptr<at::cuda::CUDAStream> stream;
    std::shared_ptr<torch::jit::script::Module> module;
    std::shared_ptr<std::vector<torch::jit::IValue>> inputs;
public:
    Launch(std::shared_ptr<at::cuda::CUDAStream> stm, std::shared_ptr<torch::jit::script::Module> mod, std::shared_ptr<std::vector<torch::jit::IValue>> inp) :
        stream{stm}, module{mod}, inputs{inp}
    {}

    void operator() (const blocked_range<size_t>& r ) const {
        for( size_t i=r.begin(); i!=r.end(); ++i ){
            at::cuda::CUDAStreamGuard torch_guard(*stream);
            module->forward(*inputs);
        }
    }
};


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
        model1 = std::make_shared<torch::jit::script::Module>(torch::jit::load("./seq_block.pth"));
        model2 = std::make_shared<torch::jit::script::Module>(torch::jit::load("./seq_block.pth"));
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
    in.push_back(torch::ones({64, 64}, tensor_options));
    auto inputs = std::make_shared<std::vector<torch::jit::IValue>>(in);
    model1->to(at::kCUDA);
    model2->to(at::kCUDA);

    std::shared_ptr<at::cuda::CUDAStream> torch_stream1 = std::make_shared<at::cuda::CUDAStream>(at::cuda::getStreamFromPool());
    std::shared_ptr<at::cuda::CUDAStream> torch_stream2 = std::make_shared<at::cuda::CUDAStream>(at::cuda::getStreamFromPool());
    tbb::task_scheduler_init init(2);
    tbb::task_group group;
    /*
    //Warm up
    for(int i=0; i<2; ++i){
        model1->forward(*inputs);
        model2->forward(*inputs);
    }
    tbb::parallel_for(0, 10, [&](int i){
            model1->forward(*inputs);
            model2->forward(*inputs);
            }
            );
            */
    for(int i=0; i<10; ++i){
        //group.run(launch(torch_stream1, model1, inputs));
        group.run([&](){
                at::cuda::CUDAStreamGuard torch_guard(*torch_stream1);
                model1->forward(*inputs);
                });
        group.run([&](){
                at::cuda::CUDAStreamGuard torch_guard(*torch_stream2);
                model2->forward(*inputs);
                });

    }
    group.wait();

    cudaDeviceSynchronize();

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end ;
    std::chrono::duration<double> span;

    int num_iter=1000;
    //tbb::parallel_for(blocked_range<size_t>(0, num_iter), Launch(torch_stream1, model1, inputs));
    //tbb::parallel_for(blocked_range<size_t>(0, num_iter), Launch(torch_stream2, model2, inputs));
    //tbb::parallel_for(0, num_iter, [&](int i){
    start = std::chrono::high_resolution_clock::now();
    for(int i=0; i<num_iter; ++i){
        group.run([&](){
                at::cuda::CUDAStreamGuard torch_guard(*torch_stream1);
                model1->forward(*inputs);
                });
        group.run([&](){
                at::cuda::CUDAStreamGuard torch_guard(*torch_stream2);
                model2->forward(*inputs);
                });

    }
    group.wait();
    
    cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();
    span = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Time per Iteration: " << (span.count()*1000)/num_iter << "ms" << std::endl;

    std::cout << "ok\n";
}

