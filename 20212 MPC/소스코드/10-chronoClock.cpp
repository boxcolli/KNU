#include <stdio.h>
#include <chrono>
#include <time.h>
using namespace std;
using namespace std::chrono;

// dummy big job
void bigjob()
{
    int count=0;    
    for (int i=0; i < 10000; ++i)
        for (int j=0; j < 10000; ++j)
            ++count;    
    printf("we got %d counts.\n", count);
}

int main()
{
    system_clock::time_point chrono_begin = system_clock::now();
    clock_t clock_begin = clock();

    bigjob();

    system_clock::time_point chrono_end = system_clock::now();
    clock_t clock_end = clock();

    long clock_elapsed_usec = (long)(clock_end-clock_begin)*1000000/CLOCKS_PER_SEC;
    printf("elapsed CPU time = %ld usec\n", clock_elapsed_usec);

    return 0;
}