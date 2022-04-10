#include "globals.h"
#include "scan.h"

int main() {
    ifstream fin("1.c");
    Scanner scanner(&fin);
    int tresult;

    while (true) {
        tresult = scanner.processChar();
        if (tresult == scanner.tERROR) {
            cout << "tresult(" << tresult << ") buffer\"" << scanner.getToken() << "\"\n";
            break;
        }
        else if (tresult != scanner.tNULL) {
            cout << "tresult(" << tresult << ") buffer\"" << scanner.getToken() << "\"\n";
        }
    }
    cout << scanner.getErrorType() << endl;
    
    
    return 0;
}