[An Easy Introduction to CUDA C and C++](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)

[Six Ways to SAXPY](https://developer.nvidia.com/blog/six-ways-saxpy/)

cuda 版本和支持的 g++(gcc) 版本要一致
- cuda 11.4 -> gcc < 11，试过，10可 `nvcc --compiler-bindir /usr/bin/g++-10 saxpy.cu -o out/saxpy`
- cuda 12.4 -> vscode 2022 可
