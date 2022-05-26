#include "globals.h"
#include "scan.h"
#include "parse.h"

int main() {
    ifstream fcode("resources/3.c");
    if (fcode.is_open()) {
        cout << "open" << endl;
    }

    //Scanner scanner(fcode);
    //while (scanner.processToken() != TokenType::tERROR) {
    //    cout << scanner.getToken() << endl;
    //}

    ofstream fout("resources/parsed.txt");
    RDParser parser(fcode, fout);
    cout << "parse end" << endl;
    parser.getTree()->show(fout);
    
    return 0;
}