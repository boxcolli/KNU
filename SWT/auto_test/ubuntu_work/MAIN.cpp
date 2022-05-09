#include "GLOBAL.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "usage: gendriver [options] <SUT file> <case file> <output name>" << endl;
        exit(1);
    }

    // pass option, open files
    int argcount = 1;
    for (; argcount != argc; argcount++) {
        if (argv[argcount][0] == '-') {
            // TODO: handle option
        }
        else {
            break;
        }
    }
    if (argcount == argc) {
        cout << "usage: gendriver [options] <SUT file> <case file> <output name>" << endl;
        exit(1);
    }
    ifstream fcode(argv[argcount++]);
    if (!fcode.is_open()) { cout << "SUT file not open" << endl; exit(1); }
    ifstream fcase(argv[argcount++]);
    if (!fcase.is_open()) { cout << "case file not open" << endl; exit(1); }
    ofstream fdrive(argv[argcount]);
    if (!fdrive.is_open()) { cout << "drive file not open" << endl; exit(1); }



    // process SUT file
    CodeScanner codeScanner(fcode);

    // process case file
    

    return 0;
}