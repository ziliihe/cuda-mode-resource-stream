#include <stdio.h>

__global__ 
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    // 两对数组
    float *x, *y, *d_x, *d_y;
    // x y 指向CPU 内存数组，使用 malloc 分配内存
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));
    // d_x,d_y 指向GPU 内存数组，使用 cudaMalloc 分配内存，cudaMalloc 是 Cuda 运行时 API
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // 初始化 x, y CPU 内存数组
    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    // 将 x, y 拷贝到 GPU 内存数组以初始化
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);  // 第 4 个参数指明拷贝方向，cudaMemcpyHostToDevice： CPU -> GPU; cudaMemcpyDeviceToHost： GPU -> CPU
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY on 1M elements
    // 网格中线程块的数量
    // 线程块中线程的数量
    saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = max(maxError, abs(y[i] - 4.0f));
    printf("Max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
}