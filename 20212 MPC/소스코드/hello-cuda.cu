#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <stdio.h>

__global__ void hello()
{
    printf("hello CUDA #%d!\n", threadIdx.x);
}

int main()
{
    hello<<<1,8>>>();
#if defined(__linux__)
    cudaDeviceSynchronize();
#endif
    return 0;
}