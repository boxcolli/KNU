#include "globals.h"
#include "scan.h"

#include <list>

typedef union ScanResult {
    TokenType ttype;
    TokErrType etype;
};

int main() {
    ifstream f_scan("1.c");
    Scanner scanner(&f_scan);
    TokenType tresult;
    
    ifstream f_print("1.c");
    string buffer;
    if (!(getline(f_print, buffer))) {
        exit(1);
    }
    list<ScanResult, string> linetokens;

    bool loop = true;

    while (loop) {
        // process one char
        tresult = scanner.processChar();

        // check token result
        if (tresult == TokenType::tNULL) {
            // do nothing
        }
        else if (tresult == TokenType::tERROR) {
            TokErrType e = scanner.getErrorType();
            if (e == TokErrType::eENDOFFILE) {
                // print
                break;
            }
        }
        else {
            linetokens.push_back()
        }
        if (scanner.isNewLine()) {
           
        }
    }
    

    
    return 0;
}