#include <iostream>
#include <fstream>

#define INF -1
#define NIL -1
#define WHITE 0
#define GRAY 1
#define BLACK 2

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



struct _node_head* make_adj_list(int vertices, int edges, ifstream& input)
{
	struct _node_head* G = new struct _node_head[vertices + 1];
	struct _node* node;
	int v1, v2;

	// initialize list
	for (int i = 1; i <= vertices; i++)
	{
		G[i].first = NULL;
		G[i].last = NULL;
	}

	// read edges
	for (int i = 1; i <= edges; i++)
	{
		input >> v1 >> v2;
		node = new struct _node;
		node->v = v2;
		node->w = NIL;
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

void dfs_tos(int vertices, struct _node_head* G)
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

	//print
	cout << "dfs p[i] :";
	for (int i = 1; i <= vertices; i++)
		cout << " " << p[i];
	cout << endl;

	cout << "tos :";
	for (struct _node* temp = list; temp != NULL; temp = temp->next)
		cout << " " << temp->v;
	cout << endl;
}

struct _node_head* make_queue()
{
	struct _node_head* temp = new struct _node_head;
	temp->first = NULL;
	temp->last = NULL;
	return temp;
}

void enqueue(int v, struct _node_head* Q)
{
	struct _node* temp = new struct _node;
	temp->v = v;
	temp->next = NULL;

	// if only 1 element
	if (Q->first == NULL)
	{
		Q->first = temp;
		Q->last = temp;
	}		
	else
	{
		// push on last
		Q->last->next = temp;
		Q->last = temp;
	}
}

struct _node* dequeue(struct _node_head* Q)
{
	// if empty
	if (Q->first == NULL)
		return NULL;

	// pop on first
	struct _node* temp = Q->first;
	Q->first = Q->first->next;

	// if only 1 element
	if (Q->first == NULL)
		Q->last = NULL;

	return temp;
}

void bfs(int vertices, struct _node_head* G)
{
	int* color = new int[vertices + 1];
	int* d = new int[vertices + 1];
	int* p = new int[vertices + 1];

	// initialize
	color[1] = GRAY;
	d[1] = 0;
	p[1] = NIL;
	for (int i = 2; i <= vertices; i++)
	{
		color[i] = WHITE;
		d[i] = INF;
		p[i] = NIL;
	}

	struct _node_head* Q = make_queue();
	struct _node* temp;
	int u, v;
	enqueue(1, Q);

	// bfs
	while ((temp = dequeue(Q)) != NULL)
	{
		u = temp->v;
		free(temp);

		for (temp = G[u].first; temp != NULL; temp = temp->next)
		{
			v = temp->v;
			if (color[v] == WHITE)
			{
				color[v] = GRAY;
				d[v] = d[u] + 1;
				p[v] = u;
				enqueue(v, Q);
			}
		}

		color[u] = BLACK;
	}

	//print
	cout << "bfs p[i] :";
	for (int i = 1; i <= vertices; i++)
		cout << " " << p[i];
	cout << endl;
}

int main()
{
	int vertices, edges;
	struct _node_head* G;

	ifstream input("input.txt");

	input >> vertices >> edges;

	G = make_adj_list(vertices, edges, input);

	print_adj_list(vertices, G);

	dfs_tos(vertices, G);

	bfs(vertices, G);
}