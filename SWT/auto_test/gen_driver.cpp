#include <iostream>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        return 0;
    }
    ifstream fcode(argv[1]);
    ifstream fcase(argv[2]);

    // check argument, return type
    string fun;
    getline(fcode, fun);
    // TODO: if not valid function, exit

    
}