/*
5 7 1
8 19 1
5 10 1
11 5 1
17 11 1
15 8 2
3 10 2
10 9 2
3 6 3
7 4 3
1 19 3
8 16 4
10 15 4
2 15 5
7 18 7
2 12 7
13 14 8
13 16 9
1 20 12
*/

#include <iostream>
#include <fstream>
#include <string>

#define HeapSize 100

using namespace std;

class Sets
{
public:
	Sets(int sz = HeapSize)
	{
		n = sz;
		parent = new int[sz];
		for (int i = 0; i < n; i++)
			parent[i] = -1;
	}
	void weightedUnion(int i, int j)
	{
		int temp = parent[i] + parent[j];
		if (parent[i] > parent[j])
		{
			parent[i] = j;
			parent[j] = temp;
		}
		else
		{
			parent[j] = i;
			parent[i] = temp;
		}
	}
	int collapsingFind(int i)
	{
		int r; // root

		for (r = i; parent[r] >= 0; r = parent[r])
			;

		while (i != r)
		{
			int s = parent[i];
			parent[i] = r;
			i = s;
		}

		return r; // return -1 if none
	}
private:
	int* parent;
	int n;
};

void kruskal(int n, int m, std::ifstream& E, Sets F)
{
	// n: vertices
	// m: edges
	int i, j;	// index (next edge)
	int p, q;	// set pointer
	int w;		// weight
	int size = 0;

	while (E >> i >> j >> w)
	{
		p = F.collapsingFind(i);
		q = F.collapsingFind(j);
		if (p != q)
		{
			F.weightedUnion(p, q);
			std::cout << i << " " << j << " " << w << endl;
			size++;

			if (size == n - 1)
				break;
		}
	}
}

int main()
{
	int vertices, edges;
	Sets F;

	std::ifstream inputfile("input.txt");

	if (inputfile.is_open())
	{
		std::cout << "file open" << endl;
		inputfile >> vertices >> edges;
		kruskal(vertices, edges, inputfile, F);
	}
	else
	{
		cout << "file fail" << endl;
	}
}
