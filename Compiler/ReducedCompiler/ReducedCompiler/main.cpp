#include "globals.h"
#include "scan.h"
#include "parse.h"

int main() {
    ifstream fcode("resources/1.c");
    if (fcode.is_open()) {
        cout << "open" << endl;
    }

    /*Scanner scanner(fcode);
    while (scanner.processToken()!=TokenType::tERROR) {
        cout << scanner.getToken() << endl;
    }*/

    RDParser parser(fcode);
    parser.getTree()->show(cout);

    /*TreeNode* t = new TreeNode(NodeKind::Decl, 0);
    t->decl = DeclAttr();*/
    

    
    

    return 0;
}