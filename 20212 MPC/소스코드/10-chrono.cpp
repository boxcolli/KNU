#include <stdio.h>
#include <time.h>
#include <chrono>
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
    system_clock::time_point chrono_begin=system_clock::now();

    bigjob();

    system_clock::time_point chrono_end=system_clock::now();

    microseconds elapsed_usec=duration_cast<microseconds>(chrono_end-chrono_begin);

    printf("elapsed time = %ld usec\n", (long)elapsed_usec.count());

    return 0;
}