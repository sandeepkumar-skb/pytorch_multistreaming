### Introduction
Our model has 2 submodels that can be launched on different CUDA Streams to improve the performance. We can launch the submodels on different CUDA streams using the Python API (context manager) but this is not exportable to TorchScript which is importing for serving(inference). So, the approach here is to create a custom op to launch the submodules on separate streams in C++/TorchScript and then use those custom-op as plugins in the original model.
 
**Steps:**
 1. Export the model to .pth file
 2. Create a Custom Op to load the .pth file, extract the submodels and launch them on different CUDA streams. Return the output tensor.
 3. This OP is then converted to a torch library using CMake.
 4. This library is loaded into the original model and will used instead of the raw python submodels.
 5. Using custom op library allows the model to be exportable with the plugins. If we had created a C++ extension as demo'ed in [pytorch_cpp_multistream](https://github.com/sandeepkumar-skb/pytorch_multistreaming/tree/main/pytorch_cpp_multistream) experiments, then we will not be able export the model.
