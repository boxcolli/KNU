#ifndef _PARSE_H_
#define _PARSE_H_

#include "globals.h"
#include "scan.h"

/**************************************************
Data
**************************************************/

enum class NodeKind {
    VarDecl, FunDecl,
    ParamList, Param,
    Stmt,
    LocalDecls, StmtList,
    Expr,
};
enum class DeclKind { VarK, FunK } ;
typedef enum {VoidK, ListK} ParamKind;
typedef enum {ExpK, CompK, SelecK, IterK, RetK} StmtKind;
typedef enum {AsgnK, OpK,ConstK,IdK} ExpKind;

/* ExpType is used for type checking */
typedef enum {Void,Integer,Boolean} ExpType;

typedef struct treeNode {
    vector<treeNode*> child;
    treeNode* sibling;
    int lineno;
    NodeKind nodekind;
    union {
        DeclKind decl;
        StmtKind stmt;
        ExpKind exp; } kind;
    union { TokenType op;
            int val;
            char * name; } attr;
    ExpType type; /* for type checking of exps */
} TreeNode;

/**************************************************
Parser
**************************************************/
class RDParser {
public:
    RDParser(ifstream& fcode);
    TreeNode* getTree() { return root; }
private:
    TreeNode* root;
    Scanner scanner;
    TokenType token;    

    void syntaxError(string message);
    void match(TokenType expected);    
    TokenType getToken();
    TreeNode* newNode(NodeKind nodeKind);
    /*************************************************/
    TreeNode* declaration_list();
    TreeNode* declaration();
    TreeNode* var_declaration();
    TreeNode* type_specifier();
    TreeNode* fun_declaration();
    TreeNode* params();
    TreeNode* param_list();
    TreeNode* param();
    TreeNode* compound_stmt();
    TreeNode* local_declaration();
    TreeNode* statement_list();
    TreeNode* statement();
    TreeNode* expression_stmt();
    TreeNode* selection_stmt();
    TreeNode* iteration_stmt();
    TreeNode* return_stmt();
    TreeNode* expression();
    TreeNode* var();
    TreeNode* simple_expression();
    TreeNode* relop();
    TreeNode* additive_expression();
    TreeNode* addop();
    TreeNode* term();
    TreeNode* mulop();
    TreeNode* factor();
    TreeNode* call();
    TreeNode* args();
    TreeNode* arg_list();
};

#endif