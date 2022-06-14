/*
input:
5 7
1 2 1
1 3 3
2 3 3
2 4 6
3 4 4
3 5 2
4 5 5

p: 1 1 3 3
d: -1 -1 4 -1
*/

#include <iostream>
#include <fstream>
#include <limits.h>

#define INF	INT_MAX
#define NIL	-1

using namespace std;

struct _edge_node
{
	int v;
	int w;
	struct _edge_node* next;
	_edge_node(int v, int w) : v(v), w(w) { next = nullptr; };
	_edge_node(int v) : v(v), w(1) { next = nullptr; };
};

struct _edge_head
{
	struct _edge_node* first;
	struct _edge_node* last;
	_edge_head() : first(nullptr), last(nullptr) {};
};

class Graph
{
public:
	int vertices;
	int edges;
	struct _edge_head* list;

	Graph(int vertices) : vertices(vertices), edges(0)
	{
		list = new struct _edge_head[vertices + 1];
	}

	void add(int u, int v, int w)
	{
		struct _edge_node* temp = new struct _edge_node(v, w);
		if (list[u].first == nullptr)
		{
			list[u].first = temp;
			list[u].last = temp;
		}
		else
		{
			list[u].last->next = temp;
			list[u].last = temp;
		}
		++edges;
	}
};

struct element
{
	int key;
	int value;
	struct element() : key(0), value(0) {};
	struct element(int k, int v) : key(k), value(v) {};
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
		list = new struct element[msize + 1];
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
		struct element item, temp;
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

	void print()
	{
		for (int i = 1; i <= size; ++i)
			cout << list[i].key << " ";
		cout << endl;
	}
};

void print_int_array(const char* title, int* ary, int a, int b)
{
	cout << title;
	for (int i = a; i <= b; ++i)
		cout << " " << ary[i];
	cout << endl;
}

void mst_prim(Graph G, int r)
{
	int* d = new int[G.vertices + 1];
	int* p = new int[G.vertices + 1];
	int* flag = new int[G.vertices + 1];
	for (int i = 0; i <= G.vertices; ++i)
	{
		d[i] = INF; p[i] = NIL; flag[i] = 0;
	}
	d[r] = 0;
	p[r] = NIL;

	Heap H(G.vertices);
	struct element e(d[r], r);
	H.insert_minheap(e);
	while (H.size > 0)
	{
		H.print();
		e = H.delete_minheap();
		int u = e.value;

		if (flag[u] == 0)
		{
			struct _edge_node* temp = G.list[u].first;
			for (; temp != nullptr; temp = temp->next)
			{
				int v = temp->v;
				int w = temp->w;
				if (flag[v] == 0 && w < d[v])
				{
					p[v] = u;
					d[v] = w;
					e.key = w; e.value = v;
					H.insert_minheap(e);
				}
 			}
			flag[u] = 1;
		}
	}

	cout << endl;
	print_int_array("parent :", p, 2, G.vertices);
	int wsum = 0;
	for (int i = 1; i <= G.vertices; ++i)
		wsum += d[i];
	cout << "weight 합 : " << wsum << endl;
	
}

void print_adj_list(int vertices, Graph G)
{
	struct _edge_node* temp;

	cout << "<adj list>" << endl;

	for (int i = 1; i <= vertices; i++)
	{
		cout << i << " :";

		temp = G.list[i].first;
		while (temp != NULL)
		{
			cout << " (" << temp->v << ", " << temp->w << ")";
			temp = temp->next;
		}
		cout << endl;
	}
	cout << "count : " << G.edges << endl;
}

int main()
{
	ifstream input("input.txt");
	
	// input: (vertices) (edges)
	int V, E;
	input >> V >> E;
	
	// input: (u) (v) (w)
	Graph G(V);
	for (int i = 0; i < E; ++i)
	{
		int u, v, w;
		input >> u >> v >> w;
 		G.add(u, v, w);
		G.add(v, u, w);
 	}

	print_adj_list(G.vertices, G);
	cout << endl;

	mst_prim(G, 1);
}