#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float randf(float A, float B, int digit)
{
    int i;
    float result;

    // confirm A < B
    if (A > B) {
        result = A; // using result as temp
        A = B;
        B = result;
    }
    // make floats big
    for(i = 0; i < digit; i++) {
        A *= 10;
        B *= 10;        
    }
    // choose rand from range(min, max)
    if (B > 0.0)
        result = (float) ( rand() % ((int)B - (int)A + 1) + (int)A );
    else {
        result = (float) ( rand() % -((int)B - (int)A + 1) + (int)A );
    }
    // make result small
    for(i = 0; i < digit; i++)
        result /= 10;
    return result;
}

int main()
{
    int i;
    printf("randf -1, -2\n");
    srand(time(NULL));
    for(i = 0; i < 10; i++)
        printf("%2d: %f\n", i, randf(-1, -2, 4));
}