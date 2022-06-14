/*
INPUT:
6
111101
110011
100001
111101
100001
100001


OUTPUT:
start : 20 = max :20
escape success!!
*/

#include <iostream>
#include <fstream>
#include <limits>
#define WHITE	0
#define GRAY	1
#define BLACK	2
#define NIL		-1
#define INF		INT_MAX

#define START	1
#define EMPTY	0
#define VIDX(x, y, n) (n*(x-1)+y)
#define VIN(idx)	(2 * idx - 1)
#define VOUT(idx)	(2 * idx)
#define VRIGHT(idx)	(idx + 1)
#define VDOWN(idx, n)	(idx + n)

using namespace std;

int vertices, edges;
int N;
int* map;

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

void checkAndAdd(Graph G, int idx1, int idx2)
{
	if (map[idx1] != START && map[idx2] != START)
	{
		G.add(VOUT(idx1), VIN(idx2), 1);
		G.add(VOUT(idx2), VIN(idx1), 1);
	}
	else if (map[idx1] == START && map[idx2] != START)
		G.add(VOUT(idx1), VIN(idx2), 1);
	else if (map[idx1] != START && map[idx2] == START)
		G.add(VOUT(idx2), VIN(idx1), 1);
}

int main()
{
	ifstream input("input.txt");

	if (input.is_open() == false)
	{
		cout << "file \"input.txt\" not found" << endl;
		exit(1);
	}

	// input: (map size)
	input >> N;
	map = new int[N * N + 1];
	vertices = 2 * N * N + 2;
	Graph G(vertices);

	// input: (START or EMPTY)
	char* buf = new char[N + 1];
	int src = 0;
	int sink = 2 * N * N + 1;
	int idx;
	int start = 0;
	for (int i = 1; i <= N; ++i)
	{
		input >> buf;
		for (int j = 1; j <= N; ++j)
		{
			idx = VIDX(i, j, N);
			map[idx] = buf[j - 1] - '0';
			G.add(VIN(idx), VOUT(idx), 1);	// v in -> v out
			if (map[idx] == START)
			{
				++start;
				G.add(src, VIN(idx), 1); // source -> START
			}
				
		}
	}
	delete[] buf;

	// 1,1
	idx = 1;
	G.add(VOUT(idx), sink, 1);			// border -> sink
	checkAndAdd(G, idx, VRIGHT(idx));	// current -> right
	checkAndAdd(G, idx, VDOWN(idx, N)); // current -> down

	// 1, [2..N-1]
	for (int j = 2; j < N; j++)
	{
		idx = j;
		G.add(VOUT(idx), sink, 1);
		checkAndAdd(G, idx, VRIGHT(idx));
		checkAndAdd(G, idx, VDOWN(idx, N));
	}

	// 1,N
	idx = N;
	G.add(VOUT(idx), sink, 1);
	checkAndAdd(G, idx, VDOWN(idx, N));

	// [2..N-1], [1..N]
	for (int i = 2; i < N; ++i)
	{
		idx = VIDX(i, 1, N);
		G.add(VOUT(idx), sink, 1);
		checkAndAdd(G, idx, VRIGHT(idx));
		checkAndAdd(G, idx, VDOWN(idx, N));

		for (int j = 2; j < N; ++j)
		{
			idx = VIDX(i, j, N);
			checkAndAdd(G, idx, VRIGHT(idx));
			checkAndAdd(G, idx, VDOWN(idx, N));
		}

		idx = VIDX(i, N, N);
		G.add(VOUT(idx), sink, 1);
		checkAndAdd(G, idx, VDOWN(idx, N));
	}

	// N,1
	idx = VIDX(N, 1, N);
	G.add(VOUT(idx), sink, 1);
	checkAndAdd(G, idx, VRIGHT(idx));

	// N, [2..N-1]
	for (int j = 2; j < N; j++)
	{
		idx = VIDX(N, j, N);
		G.add(VOUT(idx), sink, 1);
		checkAndAdd(G, idx, VRIGHT(idx));
	}

	// N,N
	idx = VIDX(N, N, N);
	G.add(VOUT(idx), sink, 1);

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
	cout << endl << "start : " << start << " = max : " << maxflow << endl;
	if (start == maxflow)
		cout << "escape success!!" << endl;
	else
		cout << "escape fail!" << endl;
	return 0;
}