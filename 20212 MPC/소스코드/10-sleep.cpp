#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

int main()
{
    system_clock::time_point chrono_begin = system_clock::now();
    clock_t clock_begin = clock();

#if defined(__linux__)
    usleep(100 * 1000); // 100 msec
#else
    Sleep(100);         // 100 msec
#endif

    system_clock::time_point chrono_end = system_clock::now();
    clock_t clock_end = clock();

    microseconds chrono_elapsed_usec = duration_cast<microseconds>(chrono_end-chrono_begin);
    printf("elapsed time = %ld usec\n", (long) chrono_elapsed_usec.count());
    long clock_elapsed_usec = (long)(clock_end-clock_begin)*1000000/CLOCKS_PER_SEC;
    printf("elapsed CPU time = %ld usec\n", clock_elapsed_usec);
    
    return 0;
}