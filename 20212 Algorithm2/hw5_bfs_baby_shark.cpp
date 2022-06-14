#include <iostream>
#include <limits.h>

#define INF INT_MAX
#define NIL -1
#define EMPTY   0
#define SHARK   9

#define WHITE   0
#define GRAY    1
#define BLACK   2

using namespace std;

class ItemBFS 
{
public:
    int x;
    int y;
    int d;

    ItemBFS(): x(-1), y(-1), d(-1) {};
    ~ItemBFS() {};
};

template <typename T>
class Node
{
public:
    T value;
    Node* next;

    Node() {};
    Node(T v, Node* n): value(v), next(n) {};
};

template <typename T>
class Queue
{
public:
    Queue(): head(nullptr), tail(nullptr), size(0) {};

    void Enqueue(T _value);
    T Dequeue();
    bool empty();

private:
    Node<T>* head;
    Node<T>* tail;
    int size;
};


int**   i2d_make(int x, int y);
void    i2d_delete(int x, int y, int** m);

ItemBFS*    fish_bfs(int N, int** G, int X, int Y, int S);
int         shark(int N, int** G, int M, int X, int Y);

int main()
{
    // N: space size
    // M: fish number
    // X, Y: current shark
    // G: space matrix
    int N, M, X, Y;
    int** G;

    cin >> N;
    M = 0;
    G = i2d_make(N, N);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            cin >> G[i][j];
            if (G[i][j] == 0)
            {
                ;
            }
            else if (G[i][j] != SHARK)
            {                
                M++;
            }
            else
            {
                X = i;
                Y = j;
            }                
        }
    
    int result = shark(N, G, M, X, Y);
    cout << "result =" << result << endl;
}

template<typename T>
void Queue<T>::Enqueue(T _value)
{
    Node<T>* node = new Node<T>;
    node->value = _value;
    size++;

    if (head == nullptr)
    {
        head = node;
        tail = node;
    }
    else
    {
        tail->next = node;
        tail = tail->next;
    }
}

template<typename T>
T Queue<T>::Dequeue()
{    
    Node<T>* node = head;
    T value = node->value;
    size--;

    if (head == tail)
    {
        head = nullptr;
        tail = nullptr;
        delete(node);
    }
    else
    {
        head = head->next;
        delete(node);            
    }

    return value;
}

template<typename T>
bool Queue<T>::empty()
{
    if (size == 0)
        return true;
    else
        return false;
}

int**   i2d_make(int x, int y)
{
    int** m = new int* [x];
    for (int i = 0; i < x; ++i)
        m[i] = new int [y];
    return m;

}
void    i2d_delete(int x, int y, int** m)
{
    for (int i = 0; i < x; ++i)
        delete(m[i]);
    delete(m);
}

ItemBFS* fish_bfs(int N, int** G, int X, int Y, int S)
{
    int** color = i2d_make(N, N);
    int** d     = i2d_make(N, N);
    //int** p     = i2d_make(N, N);

    // init
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            color[i][j] = WHITE;
            d[i][j] = INF;
            //p[i][j] = NIL;
        }
    color[X][Y] = GRAY;
    d[X][Y] = 0;
    //p[X][Y] = NIL;

    Queue<ItemBFS> Q;
    ItemBFS item;
    item.x = X;
    item.y = Y;
    item.d = 0;
    Q.Enqueue(item);

    // UP LEFT DOWN RIGHT
    int dir_x[] = {-1, 0, +1, 0};
    int dir_y[] = {0, -1, 0, +1};

    ItemBFS* fish = nullptr;

    while (Q.empty() != true)
    {
        item = Q.Dequeue();
        // x, y: current visit
        int x = item.x;
        int y = item.y;
        cout << "BFS: x=" << x << " y=" << y << endl;

        // [x][y] edible
        if (G[x][y] != 0 && G[x][y] < S)
        {
            cout << "BFS: edible" << endl;
            if (fish == nullptr)
            {
                fish = new ItemBFS;
                fish->x = x;
                fish->y = y;
                fish->d = d[x][y];
            }
            else
            {
                if (fish->d > d[x][y] ||    // shorter dist.
                    (fish->d == d[x][y] &&  // equal dist.
                        (fish->x > x || (fish->x == x && fish->y > y))))
                {
                    fish->x = x;
                    fish->y = y;
                    fish->d = d[x][y];
                }
            }
        }

        // adjacent space
        int adj_x, adj_y;
        for (int i = 0; i < 4; ++i)
        {
            //cout << "BFS: i=" << i << endl;
            adj_x = x + dir_x[i];
            adj_y = y + dir_y[i];
            
            if (adj_x >= 0 && adj_x < N && adj_y >= 0 && adj_y < N && // in G
                color[adj_x][adj_y] == WHITE && // not visited
                G[adj_x][adj_y] <= S)   // size valid
            {
                color[adj_x][adj_y] = GRAY;
                d[adj_x][adj_y] = d[x][y] + 1;

                ItemBFS temp;
                temp.x = adj_x;
                temp.y = adj_y;
                Q.Enqueue(temp);
            }
        }
    }
    cout << "BFS: finished\n\n";
    return fish;
}

int shark(int N, int** G, int M, int X, int Y)
{
    int S = 2;
    int S_count = 0;
    int time = 0;

    ItemBFS* fish;

    cout << "shark: start" << endl;

    while ((fish = fish_bfs(N, G, X, Y, S)) != nullptr)
    {
        G[X][Y] = 0;
        X = fish->x;
        Y = fish->y;
        S_count++;
        if (S_count == S)
        {
            S++;
            S_count = 0;
        }
        time += fish->d;
        G[X][Y] = 9;
        cout << "shark: X=" << X << " Y=" << Y
         << " Sc=" << S_count << " S=" << S << endl; 
    }
    
    return time;
}