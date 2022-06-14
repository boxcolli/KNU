#include "./common.cpp"

chrono::system_clock::time_point __time_begin[8] = { chrono::system_clock::now(), };

const unsigned SIZE = 256;
const float saxpy_a = 1.234f;

__global__ void kernelSAXPY(float *z, const float a, const float* x, const float* y, unsigned n)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		z[i] = fmaf(a, x[i], y[i]);
}

int main()
{
	float* vecX = new float[SIZE];
	float* vecY = new float[SIZE];
	float* vecZ = new float[SIZE];

	srand(0);
	setNormalizedRandomData(vecX, SIZE);
	setNormalizedRandomData(vecY, SIZE);

	float* dev_vecX;
	float* dev_vecY;
	float* dev_vecZ;

	cudaMalloc((void**)&dev_vecX, SIZE * sizeof(float));
	cudaMalloc((void**)&dev_vecY, SIZE * sizeof(float));
	cudaMalloc((void**)&dev_vecZ, SIZE * sizeof(float));

	ELAPSED_TIME_BEGIN(1);

	cudaMemcpy(dev_vecX, vecX, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vecY, vecY, SIZE * sizeof(float), cudaMemcpyHostToDevice);

	ELAPSED_TIME_BEGIN(0);
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((SIZE + dimBlock.x - 1) / dimBlock.x, 1, 1);
	kernelSAXPY <<<dimGrid, dimBlock >>> (dev_vecZ, saxpy_a, dev_vecX, dev_vecY, SIZE);
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);

	cudaMemcpy(vecZ, dev_vecZ, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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