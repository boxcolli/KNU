#include "globals.h"
#include "scan.h"
#include "bnf.h"
#include "parse.h"

int main() {
    ifstream fbnf("resources/bnf.txt");
    if (fbnf.is_open()) {
        cout << "open" << endl;
    }

    BNFScanner bnfScanner(&fbnf);
    
    while (bnfScanner.process()) {
        cout << bnfScanner.getToken() << " : ";
        cout << static_cast<int>(bnfScanner.getType()) << endl;
    }
    


    return 0;
}