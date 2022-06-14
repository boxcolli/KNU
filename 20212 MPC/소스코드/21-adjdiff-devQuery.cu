#include "./common.cpp"

chrono::system_clock::time_point __time_begin[8] = { chrono::system_clock::now(), };

const unsigned num = 16 * 1024 * 1024;
unsigned blocksize = 1024;

__global__ void kernelAdjDiff(float* b, const float* a, unsigned num)
{
	extern __shared__ float s_data[]; // tile
	register unsigned tx = threadIdx.x; // local index
	register unsigned i = blockIdx.x * blockDim.x + threadIdx.x; // global  index
	if (i < num)
	{
		s_data[tx] = a[i];
		__syncthreads();
		if (tx > 0)
			b[i] = s_data[tx] - s_data[tx - 1];
		else if (i > 0) // the first element of tile
			b[i] = s_data[tx] - a[i - 1];
		else // the first element of entire local memory
			b[i] = s_data[tx] - 0.0f;
	}
}

__device__ float dev_vecA[num];
__device__ float dev_vecB[num];

int main()
{
	float* vecA = new float[num];
	float* vecB = new float[num];

	srand(0);
	setNormalizedRandomData(vecA, num);
	setNormalizedRandomData(vecB, num);

	int deviceCount = 0;
	cudaDeviceProp deviceProp;
	int driverVersion = 0;
	int runtimeVersion = 0;
	cudaGetDeviceCount(&deviceCount);
	cudaGetDeviceProperties(&deviceProp, 0);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);

	ELAPSED_TIME_BEGIN(1);

	void* p_dev_vecA = nullptr;
	void* p_dev_vecB = nullptr;

	cudaGetSymbolAddress(&p_dev_vecA, dev_vecA);
	cudaGetSymbolAddress(&p_dev_vecB, dev_vecB);
	cudaMemcpy(p_dev_vecA, vecA, num * sizeof(float), cudaMemcpyHostToDevice);

	ELAPSED_TIME_BEGIN(0);
	dim3 dimBlock(1024, 1, 1);
	dim3 dimGrid((num + dimBlock.x - 1) / dimBlock.x, 1, 1);
	blocksize = deviceProp.maxThreadsPerBlock;
	kernelAdjDiff << <dimGrid, dimBlock, blocksize * sizeof(float) >> > ((float*)p_dev_vecB, (float*)p_dev_vecA, num);
	cudaDeviceSynchronize();
	ELAPSED_TIME_END(0);

	cudaMemcpy(vecB, p_dev_vecB, num * sizeof(float), cudaMemcpyDeviceToHost);

	ELAPSED_TIME_END(1);

	float sumA = getSum(vecA, num);
	float sumB = getSum(vecB, num);
	printf("SIZE = %d\n", num);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printVec("vecA", vecA, num);
	printVec("vecB", vecB, num);

	return 0;
}