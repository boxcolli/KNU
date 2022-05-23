#include "globals.h"
#include "scan.h"
#include "bnf.h"
#include "parse.h"

int main() {
    ifstream fbnf("resources/bnf.txt");
    if (fbnf.is_open()) {
        cout << "open" << endl;
    }

    ofstream fout("resources/fout.txt");

    FirstFollow ff(fbnf);
    
    fout << "first------------------------------" << endl;
    for (auto keyval : ff.firsts) {
        fout << keyval.first << ":";
        for (string s : keyval.second) {
            fout << " " << s;
        }
        fout << endl;
    }
    fout << endl;
    
    fout << "follow------------------------------" << endl;
    for (auto keyval : ff.follows) {
        fout << keyval.first << ":";
        for (string s : keyval.second) {
            fout << " " << s;
        }
        fout << endl;
    }

    return 0;
}