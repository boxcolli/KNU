#include "./common.cpp"

chrono::system_clock::time_point __time_begin[8] = { chrono::system_clock::now(), };

const unsigned SIZE = 256;
const float saxpy_a = 1.234f;
 
float host_x[SIZE];
float host_y[SIZE];
float host_z[SIZE];

__constant__ float dev_a = 1.234f;

__device__ float dev_x[SIZE];
__device__ float dev_y[SIZE];
__device__ float dev_z[SIZE];

__global__ void kernelSAXPY(unsigned n)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		dev_z[i] = fmaf(dev_a, dev_x[i], dev_y[i]);
}

int main()
{
	srand(0);
	setNormalizedRandomData(host_x, SIZE);
	setNormalizedRandomData(host_y, SIZE);

	ELAPSED_TIME_BEGIN(1);

	cudaMemcpyToSymbol(dev_x, host_x, sizeof(host_x));
	cudaMemcpyToSymbol(dev_y, host_y, sizeof(host_y));

	ELAPSED_TIME_BEGIN(0);
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((SIZE + dimBlock.x - 1) / dimBlock.x, 1, 1);
	kernelSAXPY <<<dimGrid, dimBlock >>> (SIZE);
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);

	cudaMemcpyFromSymbol(host_z, dev_z, sizeof(host_z));

	ELAPSED_TIME_END(1);

	float sumA = getSum(vecX, SIZE);
	float sumB = getSum(vecY, SIZE);
	float sumC = getSum(vecZ, SIZE);
	float diff = fabsf(sumC - (sumA + sumB));
	printf("SIZE = %d\n", SIZE);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA + sumB) = %f\n", diff);
	printf("diff(sumC, sumA + sumB) / SIZE = %f\n", diff / SIZE);
	printVec("vecA", vecX, SIZE);
	printVec("vecB", vecY, SIZE);
	printVec("vecC", vecZ, SIZE);

	return 0;
}