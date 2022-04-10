#include "globals.h"
#include "scan.h"
#include <iostream>

int main() {
    ifstream fin("1.c");
    Scanner scanner(&fin);
    int tresult = scanner.processChar();
    

    return 0;
}