#include "globals.h"
#include "tree.h"

/**************************************************
Abstract Syntax Tree
**************************************************/
DeclAttr::~DeclAttr() {
    delete this->id;
}

ParamAttr::~ParamAttr() {
    delete this->id;
}

CallAttr::~CallAttr() {
    delete this->id;
}

VarAttr::~VarAttr() {
    delete this->id;
}

NumAttr::~NumAttr() {
    delete this->str;
}

TreeNode::TreeNode(NodeKind nodeKind, int lineno) {
    this->nodeKind = nodeKind;    
    switch (nodeKind) {
    case NodeKind::Decl: // dummy type
    case NodeKind::VarDecl:
    case NodeKind::FunDecl:     decl=DeclAttr(); break;
    case NodeKind::ParamList:   paraml=ParamListAttr(); break;
    case NodeKind::Param:       param=ParamAttr(); break;
    case NodeKind::LocDecl:     ldecl=LocDeclAttr(); break;
    case NodeKind::StmtList:    slist=StmtListAttr(); break;
    case NodeKind::ExprStmt:    exps=ExpStmtAttr(); break;
    case NodeKind::CmpdStmt:    comp=ComStmtAttr(); break;
    case NodeKind::SlctStmt:    slct=SelStmtAttr(); break;
    case NodeKind::IterStmt:    iter=ItrStmtAttr(); break;
    case NodeKind::RetStmt:     ret=RetStmtAttr(); break;
    case NodeKind::Expr:        expr=ExprAttr(); break;
    case NodeKind::Oper:        
    case NodeKind::Assign:
    case NodeKind::Relop:
    case NodeKind::Addop:
    case NodeKind::Mulop:       oper=OperAttr(); break;
    case NodeKind::Call:        call=CallAttr(); break;
    case NodeKind::Var:         var=VarAttr(); break;
    case NodeKind::Type:        type=TypeAttr(); break;
    case NodeKind::Args:        args=ArgsAttr(); break;
    case NodeKind::Num:         num=NumAttr(); break;
    default:                    errt=ErrAttr(); break;
    }
    sibling=nullptr;
    this->lineno=lineno;
    this->err=ErrKind::Null;
}

string nktos(NodeKind nk) {
    switch (nk) {
    case NodeKind::Decl: return "Decl";  
    case NodeKind::VarDecl: return "VarDecl";
    case NodeKind::FunDecl: return "FunDecl";
    case NodeKind::ParamList: return "ParamList";
    case NodeKind::Param: return "Param";
    case NodeKind::LocDecl: return "LocDecl";
    case NodeKind::StmtList: return "StmtList";
    case NodeKind::ExprStmt: return "ExprStmt";
    case NodeKind::CmpdStmt: return "CmpdStmt";
    case NodeKind::SlctStmt: return "SlctStmt";
    case NodeKind::IterStmt: return "IterStmt";
    case NodeKind::RetStmt: return "RetStmt";
    case NodeKind::Expr: return "Expr";
    case NodeKind::Oper: return "Oper";
    case NodeKind::Assign: return "Assign";
    case NodeKind::Relop: return "Relop";
    case NodeKind::Addop: return "Addop";
    case NodeKind::Mulop: return "Mulop";
    case NodeKind::Call: return "Call";
    case NodeKind::Var: return "Var";
    case NodeKind::Type: return "Type";
    case NodeKind::Args: return "Args";
    case NodeKind::Num: return "Num";
    case NodeKind::ERROR: return "ERROR";
    default: return "?";
    }
}
string tktos(TypeKind tk) {
    switch (tk) {
    case TypeKind::Int: return "Int";
    case TypeKind::Void: return "void";
    case TypeKind::Err: return "Err";
    case TypeKind::Null: return "Null";
    default: return "?";
    }
}
string oktos(OperKind ok) {
    switch (ok) {
    case OperKind::Assign: return "Assign";
    case OperKind::LTE: return "LTE";
    case OperKind::LT: return "LT";
    case OperKind::GT: return "GT";
    case OperKind::GTE: return "GTE";
    case OperKind::EQ: return "EQ";
    case OperKind::NEQ: return "NEQ";
    case OperKind::ADD: return "ADD";
    case OperKind::SUB: return "SUB";
    case OperKind::MUL: return "MUL";
    case OperKind::DIV: return "DIV";
    case OperKind::Null: return "Null";
    default: return "?";
    }
}
string ektos(ErrKind ek) {
    switch (ek) {
    case ErrKind::Null: return "Null";
    case ErrKind::Err: return "Err";
    default: return "?";
    }
}

string spcheck(string* sp) {
    if (sp==nullptr)
        { return ""; }
    else
        { return *sp; }
}

void TreeNode::show(ostream& out, int level) {
    out << boolalpha;
// Step 1: print myself and my attributes
    //out << string(level*INDENTLENGTH, ' ');
    for (int i = 0; i < level; i++) {
        out << "| ";
    }    
    out << nktos(nodeKind);
    out << " :";
    //out << lineno;
switch (nodeKind) {
case NodeKind::Decl:
case NodeKind::VarDecl:
case NodeKind::FunDecl:    
    out << " [" << spcheck(decl.id) << "]";
    out << " [" << decl.ary << "]";
    break;
case NodeKind::ParamList:
    out << " [" << paraml.empty << "]";
    break;
case NodeKind::Param:
    out << " [" << spcheck(param.id) << "]";
    out << " [" << param.ary << "]";
    break;
case NodeKind::LocDecl:
    out << " [" << ldecl.empty << "]";
    break;
case NodeKind::StmtList:
    out << " [" << slist.empty << "]";
    break;
case NodeKind::ExprStmt:
    out << " [" << exps.empty << "]";
    break;
case NodeKind::CmpdStmt:
    break;
case NodeKind::SlctStmt:
    out << " [" << slct.els << "]";
    break;
case NodeKind::IterStmt:
    break;
case NodeKind::RetStmt:
    out << " [" << ret.empty << "]";
    break;
case NodeKind::Expr:
    break;
case NodeKind::Oper:    
case NodeKind::Assign:    
case NodeKind::Relop:
case NodeKind::Addop:
case NodeKind::Mulop:
    out << " [" << oktos(oper.oper) << "]";
    break;
case NodeKind::Call:
    out << " [" << spcheck(call.id) << "]";
    break;
case NodeKind::Var:
    out << " [" << spcheck(var.id) << "]";
    out << " [" << var.ary << "]";
    break;
case NodeKind::Type:
    out << " [" << tktos(type.type) << "]";
    break;
case NodeKind::Args:
    out << " [" << args.empty << "]";
    break;
case NodeKind::Num:
    out << " [" << spcheck(num.str) << "]";
    break;
default:
    out << " [?]";
    break;    
}
    out << endl;
// Step 2: print child
    for (auto c : child) {
        c->show(out, level+1);
    }
// Step 3: print sibling
    if (sibling!=nullptr) {
        sibling->show(out, level);
    }
    out << noboolalpha;
}