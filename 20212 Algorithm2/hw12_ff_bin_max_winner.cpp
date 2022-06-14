/*
10 8 5
8
6
5
3
6
4
2
7
*/

#include <iostream>
#include <fstream>
#include <limits.h>
#include <math.h>

#define INF	INT_MAX
#define NIL	-1
#define isOdd(x) (x % 2 == 1)? true : false
#define isEven(x) (x % 2 == 0)? true : false

#define NODE_LEAF	1
#define NODE_HALF	0
#define NODE_NOTLEAF	-1

using namespace std;

class MaxWinner
{
public:
	int* tree; // winner tree data
	int* list; // player data
	int n, nodes, height;
	int s, offset, lowExt;

	MaxWinner(int players, int* list) :
		list(list),
		n(players),
		nodes(n - 1),
		height(ceil(log2(n))),
		s(pow(2, (height - 1))), // size of lowest level
		offset(2 * s - 1), //  
		lowExt(2 * ((n - 1) - (s - 1))) // low players
	{
		tree = new int[nodes + 1];
		for (int i = 1; i <= nodes; ++i)
			tree[i] = NIL;

		if (n % 2 == 0) // even
		{
			for (int k = 1; k <= n; k = k + 2)
			{
				int p = getParent(k);
				tree[p] = k;
			}
		}
		else // odd
		{
			for (int k = 1; k <= lowExt; k = k + 2)
			{
				int p = (k + offset) / 2;
				tree[p] = k;
			}
			for (int k = lowExt + 2; k <= n; k = k + 2)
			{
				int p = (k - lowExt + nodes) / 2;
				tree[p] = k;
			}
		}

		int level = height - 1;
		if (level > 0)
		{
			// complete last level
			for (int i = pow(2, level - 1); i <= nodes; ++i)
				if (tree[i] == NIL)
					tree[i] = tree[i * 2];
			--level;
			// fill up
			for (; level > 0; --level)
				for (int i = pow(2, level - 1); i < pow(2, level); ++i)
					tree[i] = tree[i * 2];
		}
	}

	int getParent(int i)
	{
		if (i <= lowExt)
			return (i + offset) / 2;
		else
			return (i - lowExt + nodes) / 2;
	}

	void print()
	{
		cout << "tree:";
		for (int i = 1; i <= nodes; ++i)
			cout << " " << tree[i];
		cout << endl;
		cout << "list:";
		for (int i = 1; i <= n; ++i)
			cout << " " << list[i];
		cout << endl;
	}

	void update(int i)
	{
		// i : changed player
		int p = getParent(i);
		// update leaf node
		if (isEven(n)) // players even
		{
			// sibling : left_odd & right_even
			if (isOdd(i)) // odd
				tree[p] = (list[i] >= list[i + 1]) ? i : i + 1;
			else // even
				tree[p] = (list[i - 1] >= list[i]) ? i - 1 : i;
		}
		else // players odd
		{
			if (i <= lowExt) // low players
				if (isOdd(i))
					tree[p] = (list[i] >= list[i + 1]) ? i : i + 1;
				else // even
					tree[p] = (list[i - 1] >= list[i]) ? i - 1 : i;
			else if (i == lowExt + 1) // sib : tree[last] & i
				tree[p] = (list[tree[nodes]] >= list[i]) ? tree[nodes] : i;
			else // high players
				if (isEven(i)) // even
					tree[p] = (list[i] >= list[i + 1]) ? i : i + 1;
				else
					tree[p] = (list[i - 1] >= list[i]) ? i - 1 : i;
		}
		// update parent nodes

		p /= 2;
		for (; p > 0; p /= 2)
		{
			int child = p * 2;
			if (isEven(n))
				tree[p] = (list[tree[child]] >= list[tree[child + 1]]) ? tree[child] : tree[child + 1];
			else
			{
				if (child == nodes) // left_tree & right_player
					tree[p] = (list[tree[child]] >= list[lowExt + 1]) ? tree[child] : lowExt + 1;
				else
					tree[p] = (list[tree[child]] >= list[tree[child + 1]]) ? tree[child] : tree[child + 1];
			}
		}
	}

	int getObj(int treeIdx)
	{
		return list[tree[treeIdx]];
	}

	int getLevel(int treeIdx)
	{
		return floor(log2(treeIdx)) + 1;
	}

	int isLeaf(int treeIdx)
	{
		if (treeIdx * 2 + 1 > nodes)
			return NODE_LEAF;
		else if (treeIdx * 2 + 1 == nodes)
			return NODE_HALF;
		else
			return NODE_NOTLEAF;
	}

	void binBucket(int* objSize, int objects)
	{
		for (int i = 1; i <= objects; ++i)
		{
			// fit bucket
			int idx = 1; // tree index
			int pi;	// player index
			int obj = objSize[i]; // current object size
			bool lchild, leaf;

			//cout << "obj = " << obj << endl;

			if (isEven(n))
			{
				while (1)
				{
					if (obj <= getObj(idx))
					{
						if (2 * idx <= nodes)
						{
							idx *= 2;
							leaf = false;
						}
						else
						{
							lchild = true;
							leaf = true;
						}
							
					}
					else
					{
						if (2 * idx + 1 <= nodes)
						{
							idx = 2 * idx + 1;
							leaf = false;
						}
						else
						{
							lchild = false;
							leaf = true;
						}
					}
					
					if (leaf)
					{
						pi = tree[idx];

						if (lchild)
						{
							if (isEven(pi))
								--pi;
							if (obj <= list[pi])
								break;
							++pi;
						}
						else if (isOdd(pi))
							++pi;
						if (obj <= list[pi])
							break;

						while (isOdd(idx))
							idx /= 2;
						++idx;
					}
				}
			}
			else
			{
				while (1)
				{
					if (obj <= getObj(idx))
					{
						if (2 * idx <= nodes)
						{
							idx *= 2;
							leaf = false;
						}
						else
						{
							lchild = true;
							leaf = true;
						}

					}
					else
					{
						if (2 * idx + 1 <= nodes)
						{
							idx = 2 * idx + 1;
							leaf = false;
						}
						else
						{
							lchild = false;
							leaf = true;
						}
					}

					if (leaf)
					{
						if (idx == nodes / 2)
						{
							pi = lowExt + 1;
							if (obj <= list[pi])
								break;
						}
						pi = tree[idx];						
						if (pi <= lowExt)
						{							
							if (lchild)
							{
								if (isEven(pi))
									--pi;
								if (obj <= list[pi])
									break;
								++pi;
							}
							else if (isOdd(pi))
								++pi;
							if (obj <= list[pi])
								break;
						}
						else
						{							
							if (lchild)
							{
								if (isOdd(pi))
									--pi;
								if (obj <= list[pi])
									break;
								++pi;
							}
							else if (isEven(pi))
								++pi;
							if (obj <= list[pi])
								break;
						}

						while (isOdd(idx) || idx == nodes)
						{
							idx /= 2;
							if (idx == nodes / 2)
							{
								pi = lowExt + 1;
								if (obj <= list[pi])
									goto FIT;
							}
						}
						++idx;
					}
				}
			}
			FIT:
			// pi : fit bucket index
			list[pi] -= obj;
			update(pi);
		}


	}

	~MaxWinner()
	{
		delete[] tree;
		delete[] list;
	}
};


int main()
{
	ifstream input("input.txt");

	// input: (bin size) (objects) (bins)
	int bin_cap, objects, bins;
	input >> bin_cap >> objects >> bins;

	int* bin_list = new int[bins + 1];
	for (int i = 1; i <= bins; ++i)
		bin_list[i] = bin_cap;
	MaxWinner W(bins, bin_list);
	//W.print();

	// input: objects(object size)
	int* obj_list = new int[objects + 1];
	for (int i = 1; i <= objects; ++i)
		input >> obj_list[i];

	//cout << "start binBucket" << endl << endl;
	W.binBucket(obj_list, objects);

	cout << "bin:";
	for (int i = 1; i <= objects; ++i)
		cout << " " << obj_list[i];
	cout << endl;
}