1. QS (결과안봄)
2. Orders

소스코드, Test 결과 캡쳐
 - aabb
 - a^20
 - (a^10)b



#include <iostream>
#include <cstring>
#include <ctime>

#define MAX_INPUT 100

using namespace std;

void quicksort(char* S, int low, int high);
int rand_partition(char* S, int low, int high);
int partition(char* S, int low, int high);
int randrange(int a, int b);

void swapchar(char* a, char* b);

void perm(char* a, int k, int n);
int findbigchar(char* S, int k, int n);

int main()
{
    char* input = new char[MAX_INPUT];
    int len;
    
    std::cout << "Input string: ";
    std::cin >> input;

    len = strlen(input);

    perm(input, 0, len);

    return 0;
}



void quicksort(char* S, int low, int high)
{
    // Real quick sort algorithm

    int pivotpoint;

    if (low < high) {
        pivotpoint = rand_partition(S, low, high);
        quicksort(S, low, pivotpoint - 1);
        quicksort(S, pivotpoint + 1, high);
    }
}
int rand_partition(char* S, int low, int high)
{
    // Select pivot randomly    

    int i = randrange(low, high);

    swapchar(S + low, S + i);

    return partition(S, low, high);
}
int partition(char* S, int low, int high)
{
    // Partition
    // Initial pivot: S[low]

    int i, j, pivotpoint;
    char pivotitem;

    pivotitem = S[low];
    j = low;

    for (i = low + 1; i <= high; i++)
        if (S[i] < pivotitem)
        {
            j++;
            swapchar(S + i, S + j);
        }
    
    pivotpoint = j;
    swapchar(S + low, S + pivotpoint);
    return pivotpoint;
}

int randrange(int a, int b)
{
    srand(time(NULL));
    return rand() % (b-a+1) + a;
}

void swapchar(char* a, char* b)
{
    char temp = *a;
    *a = *b;
    *b = temp;
}

void perm(char* S, int k, int n)
{
    int m;

    if (k >= n)
    {
        std::cout << S << endl;
    }
    else
    {
        quicksort(S, k, n-1);

        perm(S, k+1, n);
        quicksort(S, k+1, n-1);

        while ((m = findbigchar(S, k, n)))
        {
            swapchar(&(S[k]), &(S[m]));
            perm(S, k+1, n);
            quicksort(S, k+1, n-1);
        }
    }
}

int findbigchar(char* S, int k, int n)
{
    // find m: (k < m && S[k] < S[m])
    int m;

    for (m = k + 1; m < n; m++)
        if (S[k] < S[m])
            return m;
    
    return 0;
}