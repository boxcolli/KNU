/*
10 15 31
1 1
2 1
2 3
3 2
3 3
3 4
4 3
4 7
4 8
4 9
4 10
5 7
5 9
5 13
5 14
6 8
6 9
6 12
7 10
7 13
7 14
7 15
8 6
8 8
8 10
9 4
9 10
9 15
10 10
10 11
10 12

Max Flow is 10
*/

#include <iostream>
#include <fstream>
#include <limits>
#define WHITE	0
#define GRAY	1
#define BLACK	2
#define NIL		-1
#define INF		INT_MAX

using namespace std;

int vertices, edges;

struct _node
{
	int v;				// vertex
	int c;				// capacity
	int f;				// flow
	struct _node* p;	// pair node
	struct _node* next;

	_node(int v, int c) : v(v), c(c), f(0)
	{
		p = nullptr;
		next = nullptr;
	}
	_node(int v, int c, struct _node* p) : v(v), c(c), f(0), p(p)
	{
		next = nullptr;
	}
};

struct _node_head
{
	struct _node* first;	// pop
	struct _node* last;		// push
};

class Graph
{
public:
	int vertices = 0;
	int edges = 0;
	struct _node_head* list;

	Graph(int vertices) : vertices(vertices)
	{
		list = new struct _node_head[vertices];
		for (int i = 0; i < vertices; ++i)
		{
			list[i].first = nullptr;
			list[i].last = nullptr;
		}
	}

	struct _node* find(int u, int v)
	{
		// find node represents edge(u, v)
		struct _node* temp = list[u].first;
		while (temp != nullptr)
		{
			if (temp->v == v)
				return temp;
			temp = temp->next;
		}
		return nullptr;
	}

	void add(int u, int v, int c)
	{
		struct _node* node = nullptr;
		struct _node* node_op = nullptr;

		struct _node* temp = find(u, v);
		if (temp == nullptr)
		{
			// add new
			node = new struct _node(v, c);
			node_op = new struct _node(u, 0, node);
			node->p = node_op;

			// add (u, v)
			if (list[u].last == nullptr)
			{
				list[u].first = node;
				list[u].last = node;
			}
			else
			{
				list[u].last->next = node;
				list[u].last = node;
			}
			// add (v, u)
			if (list[v].last == nullptr)
			{
				list[v].first = node_op;
				list[v].last = node_op;
			}
			else
			{
				list[v].last->next = node_op;
				list[v].last = node_op;
			}

			edges += 2;
		}
		else
		{
			// update c
			temp->c = c;
		}
	}

	struct _node* getList(int u)
	{
		return list[u].first;
	}
};

struct _listnode
{
	int k;
	struct _listnode* next;

	_listnode(int k) : k(k) { next = nullptr; }
	_listnode(int k, struct _listnode* next) : k(k), next(next) {}
};

class Queue
{
public:
	struct _listnode* first;
	struct _listnode* last;

	Queue() : first(nullptr), last(nullptr) {}

	void push(int k)
	{
		if (first == nullptr)
		{
			first = new struct _listnode(k);
			last = first;
		}
		else
		{
			last->next = new struct _listnode(k);
			last = last->next;
		}
	}

	int pop()
	{
		if (first == nullptr)
			return NIL;

		struct _listnode* temp = first;

		int ret = temp->k;
		if (first == last)
		{
			first = nullptr;
			last = nullptr;
		}
		else
		{
			first = first->next;
		}
		delete(temp);
		return ret;
	}

	void flush()
	{
		while (pop() != NIL)
			;
	}

	~Queue()
	{
		flush();
	}
};

Queue Q = Queue();

int bfs(Graph G, int* color, int* p, struct _node** pp)
{
	int vertices = G.vertices;
	//int* color = new int[vertices];
	//int* d = new int[vertices];
	//int* p = new int[vertices];
	//struct _node** pp = new struct _node*[vertices]; // pointer of edge from parent

	// init
	color[0] = GRAY; p[0] = NIL; pp[0] = nullptr;
	for (int i = 1; i < vertices; ++i)
	{
		color[i] = WHITE; p[i] = NIL; pp[i] = nullptr;
	}

	// bfs
	Q.flush();
	Q.push(0);
	int u, v;
	while ((u = Q.pop()) != NIL) // pop
	{
		struct _node* edge = G.getList(u);
		while (edge != nullptr) // edge (u, v)
		{
			v = edge->v;
			if (color[v] == WHITE && (edge->c - edge->f) > 0)
			{
				color[v] = GRAY;
				p[v] = u;
				pp[v] = edge;
				if (v == vertices - 1) return 0;
				Q.push(v);
			}
			edge = edge->next;
		}
		color[u] = BLACK;
	}

	return 0;
}

int maximumflow(Graph G)
{
	// Ford-Fulkersom
	int* color = new int[vertices];
	int* p = new int[vertices];
	struct _node** pp = new struct _node* [vertices];
	bfs(G, color, p, pp);
	while (p[vertices - 1] != NIL)
	{
		// find min flow on path
		int flow = pp[vertices - 1]->c - pp[vertices - 1]->f;
		for (int i = p[vertices - 1]; i > 0; i = p[i])
		{
			if ((pp[i]->c - pp[i]->f) < flow)
				flow = pp[i]->c - pp[i]->f;
		}

		// update flow
		for (int i = vertices - 1; i > 0; i = p[i])
		{
			pp[i]->f += flow;
			pp[i]->p->f = -(pp[i]->f);
		}

		bfs(G, color, p, pp);
	}

	// maxflow
	int maxflow = 0;
	struct _node* edge_sink = G.getList(vertices - 1);
	while (edge_sink != nullptr)
	{
		maxflow += edge_sink->p->f;
		edge_sink = edge_sink->next;
	}

	return maxflow;
}

int main()
{
	ifstream input("input.txt");

	if (input.is_open() == false)
	{
		cout << "file \"input.txt\" not found" << endl;
		exit(1);
	}

	// input : (elements) (elements) (edges)
	int e1, e2, edges_between;
	input >> e1 >> e2 >> edges_between;
	vertices = e1 + e2 + 2;
	int src = 0, sink = e1 + e2 + 1;
	Graph G(vertices);
	for (int i = 1; i <= e1; ++i)
		G.add(src, i, 1);			// src -> e1

	// input : (u) (v)
	for (int i = 0; i < edges_between; ++i)
	{
		int u, v;
		input >> u >> v;
		v += e1;
		G.add(u, v, 1);
	}

	for (int i = e1 + 1; i <= e1 + e2; ++i)
		G.add(i, sink, 1);			// e2 -> sink

	// print list
	cout << "총 노드 개수: " << G.edges << endl;
	for (int i = 0; i < vertices; ++i)
	{
		cout << i << ":";
		struct _node* p = G.getList(i);
		while (p != nullptr)
		{
			cout << "->(" << p->v << " " << p->c << " " << p->f << " " << p->p << ")";
			p = p->next;
		}
		cout << endl;
	}

	// maxflow
	int maxflow = maximumflow(G);
	cout << endl << "Max Flow is " << maxflow << endl;
	return 0;
}