#include "globals.h"
#include "scan.h"
#include "parse.h"

int main() {
    ifstream fcode("resources/3.c");
    if (fcode.is_open()) {
        cout << "open" << endl;
    }

    ofstream fout("resources/parsed.txt");
    RDParser parser(fcode);
    cout << "parse end" << endl;
    parser.getTree()->show(fout);
    
    return 0;
}