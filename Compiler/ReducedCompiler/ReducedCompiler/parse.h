#ifndef _PARSE_H_
#define _PARSE_H_

#include "globals.h"
#include "scan.h"

/**************************************************
Data
**************************************************/

enum class NodeKind {
    Decl, VarDecl, FunDecl,
    ParamList,
    LocDecl, StmtList,
    Stmt, ExprStmt, CmpdStmt, SlctStmt, IterStmt, RetStmt,
    Expr,
    Oper, Assign, Relop, Addop, Mulop,
    Factor, Call,
    Var, Type, Ary, Args, Num,
    ERROR
};
enum class TypeKind { Int, Void, Err, Null };
enum class OperKind { LTE, LT, GT, GTE, EQ, NEQ, ADD, SUB, MUL, DIV };
enum class ErrKind { Null, Err };

struct DeclAttr { DeclAttr() : type(TypeKind::Null), id(""), ary(false), num(0) {}
    TypeKind    type;
    string      id;
    bool        ary;
    int         num;
    ErrKind     err     = ErrKind::Null;
};
struct ParamAttr { ParamAttr() : type(TypeKind::Null), id(""), ary(false) {}
    TypeKind    type;
    string      id;
    bool        ary;
    ErrKind     err     = ErrKind::Null;
};
struct LocDeclAttr { LocDeclAttr() : empty(false) {}
    bool        empty;
    ErrKind     err     = ErrKind::Null;
};
struct StmtListAttr { StmtListAttr() : empty(false) {}
    bool        empty;
    ErrKind     err     = ErrKind::Null;
};
struct StmtAttr {
    bool        empty   = false;
    ErrKind     err     = ErrKind::Null;
};
struct CmpdAttr {

};
struct ExprAttr { ExprAttr() : err(ErrKind::Null) {}
    ErrKind     err     = ErrKind::Null;
};
struct OperAttr { OperAttr() : type(TypeKind::Null) {}
    TypeKind    type;
    OperKind    oper;
    ErrKind     err     = ErrKind::Null;
};
struct FactAttr { FactAttr() : type(TypeKind::Null), val(0) {}
    TypeKind    type;
    int         val;
    ErrKind     err     = ErrKind::Null;
};
struct CallAttr { CallAttr() : type(TypeKind::Null), id("") {}    
    TypeKind    type;
    string      id;
    ErrKind     err     = ErrKind::Null;
};
struct VarAttr { VarAttr() : type(TypeKind::Null), id("") {}
    TypeKind    type;
    string      id;
    ErrKind     err     = ErrKind::Null;
};
struct TypeAttr { TypeAttr() : type(TypeKind::Null) {}
    TypeKind    type;
    ErrKind     err     = ErrKind::Null;
};
struct ArgsAttr {
    ErrKind     err     = ErrKind::Null;
};
struct NumAttr { NumAttr() : str(""), val(0) {}
    string      str;
    int         val;
    ErrKind     err     = ErrKind::Null;
};
struct ErrAttr {};
struct TreeNode {
    NodeKind nodeKind;
    union {
        DeclAttr decl;        
        ParamAttr param;
        LocDeclAttr ldecl;
        StmtListAttr slist;
        StmtAttr stmt;
        ExprAttr expr;
        OperAttr oper;
        FactAttr fact;
        CallAttr call;
        VarAttr var;
        TypeAttr type;
        ArgsAttr args;
        NumAttr num;
        ErrAttr err; };
    vector<TreeNode*>   child;
    TreeNode*           sibling;
    int                 lineno;

    TreeNode(NodeKind nodeKind, int lineno);
};

/**************************************************
Parser
**************************************************/
class RDParser {
public:
    RDParser(ifstream& fcode);
    TreeNode* getTree() { return root; }
private:
    TreeNode*   root;
    Scanner     scanner;
    TokenType   token;
    string      tokenString;
    int         lineno;
    bool        error;

    void syntaxError(string message);
    void match(TokenType expected);    
    void nextToken();
    TreeNode* newNode(NodeKind nodeKind);
/*************************************************/
    TreeNode* declaration_list();
    TreeNode* declaration();
    TreeNode* params();
    TreeNode* compound_stmt();
    TreeNode* local_declarations();
    TreeNode* statement_list();
    TreeNode* statement();
    TreeNode* expression_stmt();
    TreeNode* compound_stmt();
    TreeNode* selection_stmt();//WIP
    TreeNode* iteration_stmt();//todo
    TreeNode* return_stmt();//todo
    TreeNode* expression();
    TreeNode* additive_expression();
    
    TreeNode* term();
    TreeNode* factor();
    TreeNode* var();
    TreeNode* call();
    TreeNode* type_specifier();
    TreeNode* args();//todo
    TreeNode* num();
};

#endif