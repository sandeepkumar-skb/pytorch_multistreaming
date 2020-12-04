#!/bin/bash

rm -rf build;
mkdir build;
cd build

#cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" ..
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..

make -j$(nproc)
