/*
4
D 0.375
I 0.375
R 0.125
W 0.125
*/

#include <iostream>
#include <fstream>
#include <math.h>

using namespace std;

typedef int keytype;
typedef int index;

struct _node
{
	keytype k;
	char name;
	_node* L;
	_node* R;
	_node() {};
	_node(int k, char name) : k(k), name(name) { L = nullptr; R = nullptr; };
	_node(int k, char name, _node* L, _node* R) : k(k), name(name), L(L), R(R) {};
};
typedef _node* node_pointer;


char* name;
int** R;


void optsearchtree(int n, const float p[], float* minavg, index** R)
{
	index i, j, k, diagonal;
	float** P = new float* [n + 1];
	float** A = new float* [n + 2];
	for (int i = 0; i <= n + 1; ++i)
	{
		P[i] = new float[n + 2];
		A[i] = new float[n + 2];
	}
	for (int i = 0; i <= n; ++i)
	{
		A[i][i - 1] = 0;
		A[i][i] = p[i];
		R[i][i] = i;
		R[i][i - 1] = 0;
		P[i][i] = p[i];
	}
	for (diagonal = 1; diagonal <= n - 1; ++diagonal)
		for (i = 1; i <= n - diagonal; ++i)
		{
			j = i + diagonal;
			P[i][j] = P[i][j-1] + P[j][j];
		}

	A[n + 1][n] = 0;
	R[n + 1][n] = 0;
	for (diagonal = 1; diagonal <= n - 1; ++diagonal)
		for (i = 1; i <= n - diagonal; ++i)
		{
			j = i + diagonal;
			k = i;
			float a = A[i][k-1] + A[k+1][j];
			for (int l = i + 1; l <= j; ++l)
			{
				float b = A[i][l - 1] + A[l + 1][j];
				if (b < a)
				{
					a = b;
					k = l;
				}
			}
			A[i][j] = a + P[i][j];
			R[i][j] = k;
		}
	*minavg = A[1][n];
}

node_pointer tree(index i, index j)
{
	index k;
	node_pointer p;
	k = R[i][j];
	if (k == 0) return NULL;
	else
	{
		p = new _node(k, name[k], tree(i, k - 1), tree(k + 1, j));
		return p;
	}
}

void preorder(node_pointer p)
{
	cout << "(" << p->name << "(";
	if (p->L)
		preorder(p->L);
	cout << ")(";
	if (p->R)
		preorder(p->R);
	cout << ")";
}

int main()
{
	ifstream input("input.txt");

	// input: (N)
	int n;
	input >> n;

	// input: (nom) (denom)
	float* p = new float[n + 1];
	name = new char[n + 1];
	for (int i = 1; i <= n; ++i)
		input >> name[i] >> p[i];


	R = new int* [n + 2];
	for (int i = 0; i <= n + 1; ++i)
		R[i] = new int[n + 2];

	float minavg;
	optsearchtree(n, p, &minavg, R);

	cout << minavg << endl;

	node_pointer root = tree(1, n);

	cout << "pre: ";
	preorder(root);
	cout << endl;

	return 0;
}