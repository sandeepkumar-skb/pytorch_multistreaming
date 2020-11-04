## To compile the torchscript:
1. Update the source file in `CMakefile.txt`
2. `./run_build.sh`

## To build torchscript as pytorch extension and import into PyTorch python API:
1. `python setup.py install` - builds and packages plugin
2. `python test_extension.py` - Test to import the plugin and run model.

## Profile torchscript using nsight-sys & nvprof
`nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o <output> -f true --cudabacktrace=true -x true ./build/run_model`
`nvprof -fo <output.nvvp> -- ./build/run_model`
