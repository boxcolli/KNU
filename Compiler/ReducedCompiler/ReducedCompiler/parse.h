#ifndef _PARSE_H_
#define _PARSE_H_

#include "globals.h"
#include "scan.h"
#include "tree.h"

/**************************************************
Parser
**************************************************/

class RDParser {
public:
    RDParser(ifstream& fcode, ostream& out);
    TreeNode* getTree() { return root; }

private:
    ifstream&   fcode;
    ostream&    out;
    Scanner     scanner;
    TreeNode*   root;
    TokenType   token;
    string      tokenString;
    int         lineno;
    bool        error;

    void syntaxError(string message);
    void match(TokenType expected);    
    void nextToken();
/*************************************************/
    TreeNode* declaration_list();
    TreeNode* declaration();
    TreeNode* params();
    TreeNode* local_declarations();
    TreeNode* statement_list();
    TreeNode* statement();
    TreeNode* expression_stmt();
    TreeNode* compound_stmt();
    TreeNode* selection_stmt();
    TreeNode* iteration_stmt();
    TreeNode* return_stmt();
    TreeNode* expression();
    TreeNode* additive_expression();
    TreeNode* term();
    TreeNode* factor();
    TreeNode* call();
    TreeNode* type_specifier();
    TreeNode* args();
    TreeNode* num();
};

#endif