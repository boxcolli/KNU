#ifndef _TREE_H_
#define _TREE_H_

#include "globals.h"

/**************************************************
Abstract Syntax Tree
**************************************************/

enum class NodeKind {
    Decl, VarDecl, FunDecl,
    ParamList, Param,
    LocDecl, StmtList,
    ExprStmt, CmpdStmt, SlctStmt, IterStmt, RetStmt,
    Expr, Oper, Assign, Relop, Addop, Mulop,    
    Var, Call, Num, Type, Ary, Args,
    ERROR
};
enum class TypeKind { Int, Void, Err, Null };
enum class OperKind { LTE, LT, GT, GTE, EQ, NEQ, ADD, SUB, MUL, DIV, Null };
enum class ErrKind { Null, Err };

struct DeclAttr { ~DeclAttr();
    string*      id     = nullptr;
    bool        ary     = false;    
};
struct ParamListAttr {
    bool        empty   = false;
};
struct ParamAttr { ~ParamAttr();
    TypeKind    type    = TypeKind::Null;
    string*      id     = nullptr;
    bool        ary     = false;
};
struct LocDeclAttr {
    bool        empty   = false;
};
struct StmtListAttr {
    bool        empty;
};
struct ExpStmtAttr {
    bool        empty   = false;
};
struct ComStmtAttr {
};
struct SelStmtAttr {
    bool        els     = false;
};
struct ItrStmtAttr {
};
struct RetStmtAttr {
    bool        empty   = false;
};
struct ExprAttr {
};
struct OperAttr {
    TypeKind    type    = TypeKind::Null;
    OperKind    oper    = OperKind::Null;
};
struct CallAttr { ~CallAttr();
    TypeKind    type    = TypeKind::Null;
    string*      id     = nullptr;
};
struct VarAttr { ~VarAttr();
    TypeKind    type    = TypeKind::Null;
    string*      id     = nullptr;
};
struct TypeAttr {
    TypeKind    type    = TypeKind::Null;
};
struct ArgsAttr {
    bool        empty   = false;
};
struct NumAttr { ~NumAttr();
    string*      str     = nullptr;
};
struct ErrAttr {};

struct TreeNode {
    NodeKind nodeKind;
    union {
        DeclAttr decl;
        ParamListAttr paraml;
        ParamAttr param;
        LocDeclAttr ldecl;
        StmtListAttr slist;
        ExpStmtAttr exps;
        ComStmtAttr comp;
        SelStmtAttr slct;
        ItrStmtAttr iter;
        RetStmtAttr ret;
        ExprAttr expr;
        OperAttr oper;
        CallAttr call;
        VarAttr var;
        TypeAttr type;
        ArgsAttr args;
        NumAttr num;
        ErrAttr errt; };
    vector<TreeNode*>   child;
    TreeNode*           sibling;
    int                 lineno;
    ErrKind             err;

    TreeNode(NodeKind nodeKind, int lineno);
    ~TreeNode() {
        while (child.size() > 0) {
            delete child.back();
            child.pop_back();
        }
        if (sibling != nullptr) {
            delete sibling;
        }
    }
    // call as default value !
    void show(ostream& out, int level=0); 
};

#endif