To compile the torchscript:
1. Update the source file in CMakefile.txt
2. ./run_build.sh

To build torchscript as pytorch extension and import into PyTorch python API:
1. python setup.py install
2. python test_extension.py - Test to import the plugin and run model.
