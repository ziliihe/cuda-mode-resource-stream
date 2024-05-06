#!/bin/bash -e
if [ ! -d "out" ];then
  mkdir out
fi
main_nvcc="nvcc --compiler-bindir /usr/bin/g++-10"

target="saxpy"
$main_nvcc $target.cu -o out/$target
echo "[ Cuda] compiled successfully!!!"
out/$target

gcc saxpy.c -o out/saxpy_c
out/saxpy_c
