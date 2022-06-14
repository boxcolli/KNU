#include <iostream>
#include <fstream>

#define HeapSize 100

using namespace std;

class Sets
{
public:
	Sets(int n)
	{
		parent = new int[n+1];	
		for (int i = 1; i <= n; i++)
		{
			parent[i] = -1;
		}
		parent[0] = n;
	}

	void weightedUnion(int i, int j, int* R)
	{
		int temp = parent[i] + parent[j];
		if (parent[i] > parent[j]) // size: i < j
		{
			parent[i] = j;
			parent[j] = temp;
		}
		else
		{
			parent[j] = i;
			parent[i] = temp;
			R[i] = R[j];
		}
	}
	int collapsingFind(int i)
	{
		int r;

		for (r = i; parent[r] >= 0; r = parent[r])
			;

		while (i != r)
		{
			int s = parent[i];
			parent[i] = r;
			i = s;
		}

		return r;
	}

private:
	int* parent;	
};

void weightedSched(int n, std::ifstream& input, int* S, int* R, Sets P)
{
	int j, d, p; // job deadline profit
	int i;

	// initialization
	
	while (input >> j >> d >> p)
	{
		// find by deadline
		i = P.collapsingFind(d);

		// schedule j
		S[R[i]] = j;

		// union deadline set
		P.weightedUnion(i, P.collapsingFind(i - 1), R);
	}
}

int main()
{
	int n;
	int* S;
	int* R;

	std::ifstream inputfile("input.txt");

	if (inputfile.is_open())
	{
		std::cout << "file open" << endl;
		inputfile >> n;

		// malloc & init
		S = new int[n + 1];
		R = new int[n + 1];
		for (int i = 1; i <= n; i++)
		{
			S[i] = 0;
			R[i] = i;
		}			
		Sets P(n);

		// run alg
		weightedSched(n, inputfile, S, R, P);

		// print
		for (int i = 1; i <= n; i++)
			std::cout << S[i] << " ";
		std::cout << endl;
	}
	else
		std::cout << "file not open" << endl;

	return 0;
}