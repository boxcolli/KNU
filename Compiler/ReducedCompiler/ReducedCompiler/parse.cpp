#include "globals.h"
#include "parse.h"
#include "scan.h"

RDParser::RDParser(ifstream& fcode) : root(nullptr), scanner(Scanner(fcode)) {
    token = getToken();
    root = declaration_list();
}
void RDParser::syntaxError(string message) {
    cout << "\n>>>";
    //cout << "Syntax error at line " << lineno
}
void RDParser::match(TokenType expected) {
    if (token == expected) token = getToken();
    else {
        syntaxError("unexpected token -> ");
    }
}
TokenType RDParser::getToken() {
    return scanner.processToken();
}
TreeNode* newNode(NodeKind nodeKind) {

}
/*************************************************/

TreeNode* RDParser::declaration_list() {
    // AST return:
    //  Decl: [ Decl: ]
    TreeNode* t = declaration();
    TreeNode* p = t;
    while ((token != TokenType::tERROR)
        && (token != TokenType::tNULL)
        && (token != TokenType::tELSE)
        && (token != TokenType::tASSIGN)
        && (token != TokenType::tCOMMA)
        && (token != TokenType::tRP)
        && (token != TokenType::tLSB)
        && (token != TokenType::tRSB)
        && (token != TokenType::tRCB)
        && (token != TokenType::tCOMMENT)) {
        TreeNode* q;
        q = declaration();
        if (q != nullptr) {
            if (t == nullptr) t = p = q;
            else {
                p->sibling = q;
                p = q;
            }
        }
    }
    return t;
}
TreeNode* RDParser::declaration() {
    // AST return:
    //  Decl:Var
    //  Decl:Fun
    TreeNode* t = type_specifier();
    
}
TreeNode* RDParser::var_declaration() {
    // AST return:
    //  Decl:Var
    //      Type Id
    //      Type Id NUM
    TreeNode* t = newNode(NodeKind::VarDecl);
}
TreeNode* RDParser::type_specifier() {
    // AST return:
    //  Type:type

}
TreeNode* RDParser::fun_declaration() {
    // AST return:
    //  Decl:Fun
    //      Type
    //      
    
}
TreeNode* RDParser::params();
TreeNode* RDParser::param_list();
TreeNode* RDParser::param();
TreeNode* RDParser::compound_stmt();
TreeNode* RDParser::local_declaration();
TreeNode* RDParser::statement_list() {
    TreeNode* t = statement();
    TreeNode* p = t;
    while ((token != TokenType::tERROR)
        && (token != TokenType::tNULL)
        && (token != TokenType::tELSE)
        && (token != TokenType::tASSIGN)
        && (token != TokenType::tCOMMA)
        && (token != TokenType::tRP)
        && (token != TokenType::tLSB)
        && (token != TokenType::tRSB)
        && (token != TokenType::tRCB)
        && (token != TokenType::tCOMMENT)) {
        TreeNode* q;
        match()
    }
}
TreeNode* RDParser::statement();
TreeNode* RDParser::expression_stmt();
TreeNode* RDParser::selection_stmt();
TreeNode* RDParser::iteration_stmt();
TreeNode* RDParser::return_stmt();
TreeNode* RDParser::expression();
TreeNode* RDParser::var();
TreeNode* RDParser::simple_expression();
TreeNode* RDParser::relop();
TreeNode* RDParser::additive_expression();
TreeNode* RDParser::addop();
TreeNode* RDParser::term();
TreeNode* RDParser::mulop();
TreeNode* RDParser::factor();
TreeNode* RDParser::call();
TreeNode* RDParser::args();
TreeNode* RDParser::arg_list();