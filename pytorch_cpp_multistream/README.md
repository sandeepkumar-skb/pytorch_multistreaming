## To compile the torchscript:
1. Update the source file in `CMakefile.txt`
2. `./run_build.sh`
RUN: `./build/run_model`

## To build torchscript as pytorch extension and import into PyTorch python API:
1. `python setup.py install` - builds and packages plugin
2. `python test_extension.py` - Test to import the plugin and run model.

## Profile torchscript using nsight-sys
```
nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o <output> -f true --cudabacktrace=true -x true ./build/run_model
```

## Profile torchscript using nvprof
```
nvprof -fo <output.nvvp> -- ./build/run_model
```

## Experiments
1. c++ threads - performance is worse than no threads. Thread creation and deletion is huge overheads.
2. p_threads with priority set - this is unfeasible in most places because setting the priority requires `sudo` priviledges
3. open_mp thread - this works really well, used the open_mp thread pool. Just using raw threads has overhead of creating and destroying threads.
4. tbb thread_pool - this works equally well as open_mp thread_pool.
