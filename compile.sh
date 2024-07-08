#!/bin/bash

if [ ! -d lib ]; then
    mkdir lib
fi

if [ ! -d build ]; then
    mkdir build
fi

cd build
cmake ..
cmake --build .
cmake --install .