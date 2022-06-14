/*
INPUT:
(vertices) (edges)
(u) (v) (w)
...
*/

#include <iostream>
#include <fstream>
#include <limits.h>

#define WHITE   0
#define GRAY    1
#define BLACK   2
#define INF     INT_MAX
#define NIL     -1

using namespace std;

struct _node {
	int v;
	int w;
	struct _node* next;
};
struct _node_head {
	struct _node* first;
	struct _node* last;
};

int vertices, edges;
int* d;
int* p;

struct _node_head* make_adj_list(int vertices, int edges, ifstream& input)
{
	struct _node_head* G = new struct _node_head[vertices + 1];
	struct _node* node;
	int v1, v2, w;

	// initialize list
	for (int i = 1; i <= vertices; i++)
	{
		G[i].first = NULL;
		G[i].last = NULL;
	}

	// read edges
	for (int i = 1; i <= edges; i++)
	{
		input >> v1 >> v2 >> w;
		node = new struct _node;
		node->v = v2;
		node->w = w;
		node->next = NULL;

		// if head node is empty
		if (G[v1].last == NULL)
		{
			G[v1].first = node;
			G[v1].last = node;
		}
		else
		{
			G[v1].last->next = node;
			G[v1].last = node;
		}
	}

	return G;
}

void print_adj_list(int vertices, struct _node_head* G)
{
	struct _node* temp;

	cout << "<adj list>" << endl;

	for (int i = 1; i <= vertices; i++)
	{
		cout << i << ":";

		temp = G[i].first;
		while (temp != NULL)
		{
			cout << " " << temp->v;
			temp = temp->next;
		}

		cout << endl;
	}
}

void dfs_tos_visit(int u, struct _node_head* G, int* color, int* p, struct _node** list)
{
	color[u] = GRAY;

	for (struct _node* temp = G[u].first; temp != NULL; temp = temp->next)
	{
		int v = temp->v;

		if (color[v] == WHITE)
		{
			p[v] = u;
			dfs_tos_visit(v, G, color, p, list);
		}
	}

	color[u] = BLACK;

	// push vertex(u)
	struct _node* temp = new struct _node;
	temp->v = u;
	temp->next = *list;
	*list = temp;
}

struct _node* dfs_tos(int vertices, struct _node_head* G)
{
	int* color = new int[vertices + 1];
	int* p = new int[vertices + 1];
	struct _node* list;

	// initialize
	for (int i = 1; i <= vertices; i++)
	{
		color[i] = WHITE;
		p[i] = NIL;
	}
	list = NULL;

	// dfs_tos
	for (int i = 1; i <= vertices; i++)
	{
		if (color[i] == WHITE)
			dfs_tos_visit(i, G, color, p, &list);
	}

	return list;
}

void initialize_single_source(struct _node_head* G, int s)
{
	d = new int[vertices + 1];
	p = new int[vertices + 1];
	for (int i = 1; i <= vertices; ++i)
	{
		d[i] = INF;
		p[i] = NIL;
	}
	d[s] = 0;
}

void relax(int u, int v, int w_uv)
{
	if (d[v] > d[u] + w_uv)
	{
		d[v] = d[u] + w_uv;
		p[v] = u;
	}
}

void bellman_ford(struct _node_head* G, int s, struct _node* tos)
{
	initialize_single_source(G, s);
	for (; tos != NULL; tos = tos->next)
	{
		int u = tos->v;
		struct _node* edge = G[u].first;
		while (edge != NULL)
		{
			relax(u, edge->v, edge->w);
			edge = edge->next;
		}
	}
}

int main()
{
	std::ifstream input("input.txt");

	if (input.is_open())
		cout << "file open" << endl;

	// input: vertices edges	
	input >> vertices >> edges;
	struct _node_head* G = make_adj_list(vertices, edges, input);

	/*
	input.clear();
	input.seekg(0, input.beg);
	input >> vertices >> edges;
	struct _node_head* W = make_adj_list(vertices, edges, input);
	*/

	struct _node* tos = dfs_tos(vertices, G);
	bellman_ford(G, 1, tos);

	cout << "d[i]: ";
	for (int i = 1; i <= vertices; ++i)
		cout << d[i] << " ";
	cout << endl;

	cout << "p[i]: ";
	for (int i = 1; i <= vertices; ++i)
		cout << p[i] << " ";
	cout << endl;
}