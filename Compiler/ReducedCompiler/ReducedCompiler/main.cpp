#include "globals.h"
#include "scan.h"
#include "bnf.h"
#include "parse.h"

int main() {
    ifstream fbnf("resources/bnf.txt");
    if (fbnf.is_open()) {
        cout << "open" << endl;
    }

    _BNFParser bnfParser(fbnf);
    grammar g = bnfParser.getGrammar();
    auto sm = bnfParser.getSymmap();
    auto ts = bnfParser.getTermset();

    cout << "grammar" << endl;
    int count = 0;
    for (auto rule : g) {
        cout << rule.l << "[" << count++ << "] :";
        for (auto tok : rule.r) {
            cout << " " << tok.second;
        }
        cout << endl;
    }
    cout << "\n\n\n";

    cout << "symbol map" << endl;
    for (auto const& keyValue : sm) {
        cout << keyValue.first << " : "
            << keyValue.second.first << " "
            << keyValue.second.second << endl;
    }
    cout << "\n\n\n";
    cout << "term set" << endl;
    for (auto item : ts) {
        cout << item << endl;
    }
    
    return 0;
}