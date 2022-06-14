//#include "./common.cpp"
#include <stdio.h>

// kernel program for the device (GPU): compiled by NVCC
__global__ void add_kernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// main program for the CPU: compiled by MS-VC++
int main()
{
    // host-side data
    const int SIZE = 5;
    const int a[SIZE] = {1, 2, 3, 4, 5};
    const int b[SIZE] = {10, 20, 30, 40, 50};
    int c[SIZE] = {0};

    // device-side data
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    
    // allocate device memory
    cudaMalloc((void**)&dev_a, SIZE*sizeof(int));
    cudaMalloc((void**)&dev_b, SIZE*sizeof(int));
    cudaMalloc((void**)&dev_c, SIZE*sizeof(int));

    // copy from host to device
    cudaMemcpy(dev_a, a, SIZE*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, SIZE*sizeof(int), cudaMemcpyHostToDevice);

    // launch a kernel on the GPU with one thread for each element.
    add_kernel<<<1, SIZE>>>(dev_c, dev_a, dev_b);
    cudaDeviceSynchronize();

    // copy from device to host
    cudaMemcpy(c, dev_c, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    // print
    printf("%d %d %d %d %d + %d %d %d %d %d = %d %d %d %d %d\n",
            a[0], a[1], a[2], a[3], a[4],
            b[0], b[1], b[2], b[3], b[4],
            c[0], c[1], c[2], c[3], c[4]);
    
    // done
    fflush(stdout);
    return 0;
}