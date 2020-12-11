#!/bin/bash

git clone https://github.com/yuki-koyama/hello-tbb-cmake.git --recursive
rm -rf build;
mkdir build;
cd build

cmake  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" ..

make -j$(nproc)
