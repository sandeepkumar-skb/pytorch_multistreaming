# PyTorch Multistreaming Experiments
PyTorch CUDA multistreaming  in Python and CPP(TorchScript)
1. [Pytorch_py_multistream](https://github.com/sandeepkumar-skb/pytorch_multistreaming/tree/main/pytorch_py_multistream) - This has examples for launching models on multiple CUDA streams 
using pytorch context managers using the Python API.
2. [PyTorch_cpp_multistream](https://github.com/sandeepkumar-skb/pytorch_multistreaming/tree/main/pytorch_cpp_multistream) - This has examples for launching models on multiple CUDA streams
using C++ TorchScript and converting it to a CPP extension which can be imported into Python model.
   This has the following examples:
   a. Launching models on CUDA streams using TorchScript.
   b. Converting them into CPP extensions which can be imported into the python model.
   c. Experiments for launching on multiple CUDA Streams using
    i. Threads
    ii. async_launch
    iii. pthreads with priority
    iv. OpenMP thread pools
    v. TBB thread pools
3. [seq_block_custom_op](https://github.com/sandeepkumar-skb/pytorch_multistreaming/tree/main/seq_block_custom_op)- This demonstrates how to launch models on different CUDA streams using CPP custom op and 
compiling it into a library which can be loaded into Python or CPP modules.
