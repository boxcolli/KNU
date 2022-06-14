/*
20
16
19
20
3
4
8
2
1
9
4
11
13
25
96
13
20
45
13
2
4  
*/

#include <iostream>
#include <fstream>

using namespace std;

struct element
{
	int key;
};

class Heap
{
public:
	int max_size;
	int size;
	struct element* list;

	Heap(int msize) : max_size(msize)
	{
		size = 0;
		list = new element[msize + 1];
	}

	void insert_maxheap(struct element item)
	{
		int i;

		for (i = ++size; i != 1 && item.key > list[i / 2].key; i /= 2)
			list[i] = list[i / 2];
		list[i] = item;
	}
	void insert_minheap(struct element item)
	{
		int i;

		for (i = ++size; i != 1 && item.key < list[i / 2].key; i /= 2)
			list[i] = list[i / 2];
		list[i] = item;
	}

	struct element delete_maxheap()
	{
		int parent, child;
		element item, temp;
		item = list[1];
		temp = list[size--];
		parent = 1;
		child = 2;
		while (child <= size) {
			// 현재 노드의 자식노드중 더 큰 자식노드를 찾는다.
			if ((child < size) &&
				(list[child].key) < list[child + 1].key)
				child++;
			if (temp.key >= list[child].key) break;
			// 한단계 아래로 이동
			list[parent] = list[child];
			parent = child;
			child *= 2;
		}
		list[parent] = temp;
		return item;
	}
	struct element delete_minheap()
	{
		int parent, child;
		element item, temp;
		item = list[1];
		temp = list[size--];
		parent = 1;
		child = 2;
		while (child <= size) {
			if ((child < size) &&
				(list[child].key) > list[child + 1].key)
				child++;
			if (temp.key <= list[child].key) break;
			list[parent] = list[child];
			parent = child;
			child *= 2;
		}
		list[parent] = temp;
		return item;
	}
};

void print_element(const char* title, struct element* list, int n)
{
	cout << title;
	for (int i = 1; i <= n; ++i)
		cout << " " << list[i].key;
	cout << endl;
}

void copy_element(struct element* dest, struct element* src, int n)
{
	for (int i = 1; i <= n; ++i)
		dest[i] = src[i];
}

int main()
{
	ifstream input("input.txt");
	if (!input.is_open())
	{
		cout << "file not open" << endl;
		exit(1);
	}

	int n;
	input >> n;
	Heap H(n * 10);

	for (int i = 0; i < n; ++i)
	{
		struct element e;
		input >> e.key;
		H.insert_minheap(e);
	}

	print_element("heap:", H.list, n);

	// sort
	struct element* minsort = new struct element[n + 1];
	struct element* temp = new struct element[n + 1];
	copy_element(temp, H.list, n);
	for (int i = 1; i <= n; ++i)
	{
		minsort[i] = H.delete_minheap();
	}

	print_element("sort:", minsort, n);
	delete[] minsort;
	delete[] H.list;
	H.list = temp;
	H.size = n;

#define CMD_DEL	-9999
#define CMD_EXIT 9999

	while (true)
	{
		int cmd;

		cout << endl << "command: ";
		cin >> cmd;

		if (cmd == CMD_EXIT) break;
		if (cmd == CMD_DEL)
		{
			// delete min element
			H.delete_minheap();
			print_element("heap:", H.list, H.size);
			cout << endl;
		}
		else
		{
			// insert new element
			struct element e;
			e.key = cmd;
			H.insert_minheap(e);
			print_element("heap:", H.list, H.size);
			cout << endl;
		}
	}

	delete[] H.list;
}