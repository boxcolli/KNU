#include "./common.cpp"

chrono::system_clock::time_point __time_begin[8] = { chrono::system_clock::now(), };

unsigned vecSize = 256 * 1024 * 1024;

__global__ void kernelVecAdd(float *c, const float* a, const float* b, unsigned n, long long* times)
{
	clock_t start = clock();
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		c[i] = a[i] + b[i];
	clock_t end = clock();
	if (i == 0)
		times[0] = (long long)(end - start);
}

int main(const int argc, const char* argv[])
{
	// argv processing
	char* pEnd = nullptr;
	switch (argc)
	{
	case 1:
		break;
	case 2:
		vecSize = strtol(argv[1], &pEnd, 10);
		break;
	default:
		printf("usage: %s [size]\n", argv[0]);
		exit(EXIT_FAILURE);
		break;
	}

	float* vecA = new float[vecSize];
	float* vecB = new float[vecSize];
	float* vecC = new float[vecSize];
	long long* host_times = new long long[1];

	srand(0);
	setNormalizedRandomData(vecA, vecSize);
	setNormalizedRandomData(vecB, vecSize);

	float* dev_vecA;
	float* dev_vecB;
	float* dev_vecC;
	long long* dev_times = nullptr;

	cudaMalloc((void**)&dev_vecA, vecSize * sizeof(float));
	cudaMalloc((void**)&dev_vecB, vecSize * sizeof(float));
	cudaMalloc((void**)&dev_vecC, vecSize * sizeof(float));
	cudaMalloc((void**)&dev_times, 1 * sizeof(long long));

	ELAPSED_TIME_BEGIN(1);

	cudaMemcpy(dev_vecA, vecA, vecSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vecB, vecB, vecSize * sizeof(float), cudaMemcpyHostToDevice);

	ELAPSED_TIME_BEGIN(0);
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((vecSize + dimBlock.x - 1) / dimBlock.x, 1, 1);
	kernelVecAdd <<<dimGrid, dimBlock >>> (dev_vecC, dev_vecA, dev_vecB, SIZE, dev_times);
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);

	cudaMemcpy(vecC, dev_vecC, vecSize * sizeof(float), cudaMemcpyDeviceToHost);
	
	ELAPSED_TIME_END(1);

	cudaMemcpy(host_times, dev_times, 1 * sizeof(long long), cudaMemcpyDeviceToHost);

	// kernel clock calculation
	int peak_clk = 1;
	cudaDeviceGetAttribute(&peak_clk, cudaDevAttrClockRate, 0);
	printf("num clock = %lld, peak clock rate = %dkHz, elapsed time: %f usec\n",
		host_times[0], peak_clk, host_times[0] * 1000.0f / (float)peak_clk);

	float sumA = getSum(vecA, vecSize);
	float sumB = getSum(vecB, vecSize);
	float sumC = getSum(vecC, vecSize);
	float diff = fabsf(sumC - (sumA + sumB));
	printf("SIZE = %d\n", vecSize);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA + sumB) = %f\n", diff);
	printf("diff(sumC, sumA + sumB) / SIZE = %f\n", diff / vecSize);
	printVec("vecA", vecA, vecSize);
	printVec("vecB", vecB, vecSize);
	printVec("vecC", vecC, vecSize);

	return 0;
}