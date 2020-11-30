### Introduction
 We have a model which has 2 submodels that can be launched on different CUDA Streams. Since can launch the submodels on different CUDA streams using the Python API (context manager) but
 this is not feasible because we want to deploy using TorchScript. Python context manager to launch on different streams are currently not exportable.
 So, here we are doing the following things:
 1. Exporting the model to .pth file
 2. Creating a Custom Op to load the .pth file, extract the submodels and launch them on different CUDA streams. It will then return the output tensor.
 3. This OP is then converted to a torch library using CMake.
 4. This library is loaded into the original model and will used instead of the raw python submodels.
 5. Advantage of creating this into a library is that this is now exportable. If we have created a C++ extension in `pytorch_cpp_multistream` experiments, then we wouldn't be able to export the model.
