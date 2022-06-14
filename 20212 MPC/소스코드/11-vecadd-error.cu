#include "./common.cpp"

const unsigned SIZE = 1024 * 1024;

__global__ void kernelVecAdd(float* c, const float* a, const float* b)
{
    unsigned i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    float* vecA = new float[SIZE];
    float* vecB = new float[SIZE];
    float* vecC = new float[SIZE];

    setNormalizedRandomData(vecA, SIZE);
    setNormalizedRandomData(vecB, SIZE);

    float* dev_vecA;
    float* dev_vecB;
    float* dev_vecC;

    cudaMalloc((void**)&dev_vecA, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_vecB, SIZE*sizeof(float));
    cudaMalloc((void**)&dev_vecC, SIZE*sizeof(float));

    ELAPSED_TIME_BEGIN(1);

    cudaMemcpy(dev_vecA, vecA, SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vecB, vecB, SIZE*sizeof(float), cudaMemcpyHostToDevice);

    ELAPSED_TIME_BEGIN(0);

    kernelVecAdd<<<1,SIZE>>>(dev_vecC, dev_vecA, dev_vecB);
    cudaDeviceSynchronize();

    ELAPSED_TIME_END(0);
 
    cudaMemcpy(vecC, dev_vecC, SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    ELAPSED_TIME_END(1);

    float sumA = getSum(vecA, SIZE);
    float sumB = getSum(vecB, SIZE);
    float sumC = getSum(vecC, SIZE);
    float diff = fabsf(sumC - (sumA + sumB));
    printf("SIZE = %d\n", SIZE);
    printf("sumA = %f\n", sumA);
    printf("sumB = %f\n", sumB);
    printf("sumC = %f\n", sumC);
    printf("diff(sumC, sumA + sumB) = %f\n", diff);
    printf("diff(sumC, sumA + sumB) / SIZE = %f\n", diff / SIZE);
    printVec("vecA", vecA, SIZE);
    printVec("vecB", vecB, SIZE);
    printVec("vecC", vecC, SIZE);

    return 0;
}