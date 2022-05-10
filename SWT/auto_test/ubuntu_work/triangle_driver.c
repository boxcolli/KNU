#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int triangle(int a, int b, int c);

int main() {
	if (triangle(2, 2, 4) == 2) printf("test case 0 : pass\n");
	else printf("test case 0 : Fail\n");

	if (triangle(3, 2, 4) == 3) printf("test case 1 : pass\n");
	else printf("test case 1 : Fail\n");

	if (triangle(3, 3, 3) == 0) printf("test case 2 : pass\n");
	else printf("test case 2 : Fail\n");

	if (triangle(1, 2, 5) == 2) printf("test case 3 : pass\n");
	else printf("test case 3 : Fail\n");

	return 0;
}