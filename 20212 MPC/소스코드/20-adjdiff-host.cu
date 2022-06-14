#include "./common.cpp"

chrono::system_clock::time_point __time_begin[8] = { chrono::system_clock::now(), };

unsigned num = 16 * 1024 * 1024;

int main()
{
	float* vecA = new float[num];
	float* vecB = new float[num];
	float* vecC = new float[num];

	srand(0);
	setNormalizedRandomData(vecA, num);
	setNormalizedRandomData(vecB, num);

	ELAPSED_TIME_BEGIN(0);
	for (register unsigned i = 0; i < num; ++i)
		if (i == 0)
			vecB[i] = vecA[i] - 0.0f;
		else
			vecB[i] = vecA[i] - vecA[i - 1];
	ELAPSED_TIME_END(0);

	float sumA = getSum(vecA, num);
	float sumB = getSum(vecB, num);
	float sumC = getSum(vecC, num);
	float diff = fabsf(sumC - (sumA + sumB));
	printf("SIZE = %d\n", num);
	printf("sumA = %f\n", sumA);
	printf("sumB = %f\n", sumB);
	printf("sumC = %f\n", sumC);
	printf("diff(sumC, sumA + sumB) = %f\n", diff);
	printf("diff(sumC, sumA + sumB) / SIZE = %f\n", diff / num);
	printVec("vecA", vecA, num);
	printVec("vecB", vecB, num);
	printVec("vecC", vecC, num);

	return 0;
}