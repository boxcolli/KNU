#include "globals.h"
#include "parse.h"
#include "scan.h"

TreeNode::TreeNode(NodeKind nodeKind, int lineno) {
    this->nodeKind = nodeKind;    
    switch (nodeKind) {
    case NodeKind::Decl: 
    case NodeKind::VarDecl:
    case NodeKind::FunDecl:     decl=DeclAttr(); break;
    case NodeKind::ParamList:   param=ParamAttr(); break;
    case NodeKind::LocDecl:     ldecl=LocDeclAttr(); break;
    case NodeKind::StmtList:    slist=StmtListAttr(); break;
    case NodeKind::Stmt:
    case NodeKind::ExprStmt:
    case NodeKind::CmpdStmt:
    case NodeKind::SlctStmt:
    case NodeKind::IterStmt:
    case NodeKind::RetStmt:     stmt=StmtAttr(); break;
    case NodeKind::Expr:        expr=ExprAttr(); break;
    case NodeKind::Assign:        
    case NodeKind::Relop:
    case NodeKind::Addop:
    case NodeKind::Mulop:       oper=OperAttr(); break;
    case NodeKind::Term:        term=TermAttr(); break;
    case NodeKind::Factor:      fact=FactAttr(); break;
    case NodeKind::Call:        call=CallAttr(); break;
    case NodeKind::Var:         var=VarAttr(); break;
    case NodeKind::Type:        type=TypeAttr(); break;
    case NodeKind::Args:        args=ArgsAttr(); break;
    case NodeKind::Num:         num=NumAttr(); break;
    default:                    err=ErrAttr(); break;
    }
    sibling=nullptr;
    this->lineno=lineno;
}

RDParser::RDParser(ifstream& fcode) : root(nullptr), scanner(Scanner(fcode)) {
    nextToken();
    root = declaration_list();
}
void RDParser::syntaxError(string message) {
    cout << "\n>>>";
    cout << "Syntax error at line " << lineno << ": " << message;
    cout << "(was: " << tokenString << ")" << endl;
    error = true;
}
void RDParser::match(TokenType expected) {
    if (token == expected) nextToken();
    else {
        syntaxError("unexpected token -> ");
    }
}
void RDParser::nextToken() {
    token = scanner.processToken();
    lineno = scanner.getLineno();
    tokenString = scanner.getToken();
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
    TreeNode* t = new TreeNode(NodeKind::Decl, lineno);
    // <type-specifier>    
    TreeNode* p = type_specifier();
    // ID
    if (token==TokenType::tID) {
        t->decl.id = tokenString;        
        nextToken();
    }
    else {
        syntaxError("unexpected token -> ");
        t->decl.id = "";        
    }
    // SEMI | LSB | LP
    switch (token) {
    case TokenType::tSEMI:
        t->nodeKind=NodeKind::VarDecl;
        match(TokenType::tSEMI);
        break;
    case TokenType::tLSB:
        t->nodeKind=NodeKind::VarDecl;
        t->decl.ary=true;
        match(TokenType::tLSB);
        t->child.push_back(num());
        match(TokenType::tRSB);
        match(TokenType::tSEMI);
        break;
    case TokenType::tLP:
        match(TokenType::tLP);
        t->nodeKind=NodeKind::FunDecl;
        t->child.push_back(params());
        match(TokenType::tRP);
        t->child.push_back(compound_stmt());
        break;
    default:
        syntaxError("unexpected token -> ");
        break;
    }
    return t;
}

TreeNode* RDParser::params() {
    TreeNode* t = new TreeNode(NodeKind::ParamList, lineno);
    TreeNode* p = t;
    if (token==TokenType::tVOID) {
        t->param.type=TypeKind::Void; }
    else {
        // <param> { COMMA <param> }
        // <type-specifier> ID [ LSB RSB ]
        t->child.push_back(type_specifier());
        if (token==TokenType::tID) {
            t->param.id=tokenString;
            nextToken();
        }
        else { syntaxError("unexpected token -> "); }
        if (token==TokenType::tLSB) {
            t->param.ary=true;
            match(TokenType::tLSB);            
            match(TokenType::tRSB);
        }
        while (token==TokenType::tCOMMA) {
            match(TokenType::tCOMMA);
            TreeNode* q = new TreeNode(NodeKind::ParamList, lineno);
            q->child.push_back(type_specifier());
            if (token==TokenType::tID) {
                q->param.id=tokenString;
                nextToken();
            }
            else { syntaxError("unexpected token -> "); }
            if (token==TokenType::tLSB) {
                t->param.ary=true;
                match(TokenType::tLSB);            
                match(TokenType::tRSB);
            }            
            p->sibling=q;
            p=q;
        }
    }
    return t;
}
TreeNode* RDParser::compound_stmt() {
    TreeNode* t = new TreeNode(NodeKind::CmpdStmt, lineno);
    match(TokenType::tLCB);
    t->child.push_back(local_declarations());
    t->child.push_back(statement_list());
    //TODO
}
TreeNode* RDParser::local_declarations() {
    TreeNode* t = new TreeNode(NodeKind::LocDecl, lineno);
    if ((token==TokenType::tINT)
        || (token==TokenType::tVOID)) {        
        // <type-specifier>
        t->child.push_back(type_specifier());
        // ID
        if (token==TokenType::tID) {
            t->decl.id=tokenString;
            nextToken();        
        }
        else { syntaxError("unexpected token -> "); }
        // SEMI | LSB
        if (token==TokenType::tLSB) {
            t->decl.ary=true;
            match(TokenType::tLSB);
            t->child.push_back(num());
            match(TokenType::tRSB);
        }
        match(TokenType::tSEMI);
        TreeNode* p = t;
        while ((token==TokenType::tINT)
            || (token==TokenType::tVOID)) {
            TreeNode* q = new TreeNode(NodeKind::LocDecl, lineno);
            // <type-specifier>
            t->child.push_back(type_specifier());
            // ID
            if (token==TokenType::tID) {
                t->decl.id=tokenString;
                nextToken();
            }
            else { syntaxError("unexpected token -> "); }
            // SEMI | LSB
            if (token==TokenType::tLSB) {
                t->decl.ary=true;
                match(TokenType::tLSB);
                t->child.push_back(num());
                match(TokenType::tRSB);
            }
            match(TokenType::tSEMI);
            p->sibling=q;
            p=q;
        }
    }
    else { t->ldecl.empty=true; }
    return t;
}
TreeNode* RDParser::statement_list() {
    TreeNode* t = new TreeNode(NodeKind::StmtList, lineno);
    if ((token==TokenType::tID)
        ||(token==TokenType::tIF)
        ||(token==TokenType::tLCB)
        ||(token==TokenType::tLP)
        ||(token==TokenType::tNUM)
        ||(token==TokenType::tRETURN)
        ||(token==TokenType::tSEMI)
        ||(token==TokenType::tWHILE)) { // firsts of <statement-list>
        TreeNode* p = statement();
        t->child.push_back(p);
        while ((token==TokenType::tID)
            ||(token==TokenType::tIF)
            ||(token==TokenType::tLCB)
            ||(token==TokenType::tLP)
            ||(token==TokenType::tNUM)
            ||(token==TokenType::tRETURN)
            ||(token==TokenType::tSEMI)
            ||(token==TokenType::tWHILE)) {
            TreeNode* q = statement();
            p->sibling=q;
            p=q;
        }
    }
    else { t->slist.empty=true; }
    return t;
}
TreeNode* RDParser::statement() {
    TreeNode* t;
    switch (token) {
    case TokenType::tID:
    case TokenType::tLP:
    case TokenType::tNUM:
    case TokenType::tSEMI:      t = expression_stmt(); break;
    case TokenType::tLCB:       t = compound_stmt(); break;
    case TokenType::tIF:        t = selection_stmt(); break;
    case TokenType::tWHILE:     t = iteration_stmt(); break;
    case TokenType::tRETURN:    t = return_stmt(); break;
    default:
        t = nullptr; break;
    }
    return t;
}
TreeNode* RDParser::expression_stmt() {
    TreeNode* t = new TreeNode(NodeKind::ExprStmt, lineno);
    if (token==TokenType::tSEMI) {        
        t->stmt.empty=true;
    }
    else { t->child.push_back(expression()); }
    match(TokenType::tSEMI);
    return t;
}
TreeNode* RDParser::expression() {    
    TreeNode* t = new TreeNode(NodeKind::Expr, lineno);
    TreeNode* p = nullptr;
    TreeNode* q = nullptr;
    string id;
    switch (token) {
    case TokenType::tID: // <var> | <call>
        // <var> [ ASSIGN <expression> ] | <call>
        id = tokenString;
        nextToken();
        switch (token) {
        case TokenType::tLSB: // <var> [ LSB <expression> RSB ]
            p = new TreeNode(NodeKind::Var, lineno);
            p->var.id=id;
            match(TokenType::tLSB);
            p->child.push_back(expression());
            match(TokenType::tRSB);
            break;
        case TokenType::tASSIGN: // <var> [ ASSIGN <expression> ]
            p = new TreeNode(NodeKind::Assign, lineno);
            q = new TreeNode(NodeKind::Var, lineno);
            q->var.id=id;
            p->child.push_back(q);
            p->child.push_back(expression());            
            break;
        case TokenType::tLP: // <call> ::= ID LP <args> RP
            p = new TreeNode(NodeKind::Call, lineno);
            p->call.id=id;
            match(TokenType::tLP);
            p->child.push_back(args());
            match(TokenType::tRP);
            break;
        default: // <var> OP?
            p = new TreeNode(NodeKind::Var, lineno);
            p->var.id=id;
            switch (token) {
            case TokenType::tLTE:
            case TokenType::tLT:
            case TokenType::tGT:
            case TokenType::tGTE:
            case TokenType::tEQ:
            case TokenType::tNEQ:
                q = p;
                p = relop();
                p->child.insert(q->child.begin(), q);
                break;
            case TokenType::tADD:
            case TokenType::tSUB:
                q = p;
                p = addop();
                p->child.insert(q->child.begin(), q);
                break;
            case TokenType::tMUL:
            case TokenType::tDIV:
                q = p;
                p = mulop();
                p->child.insert(q->child.begin(), q);
                break;
            default: // <var> only
                break;
            }
            break;
        }
        t->child.push_back(p);
        break;
    case TokenType::tLP: // <factor> ::= LP <expression> RP
        match(TokenType::tLP);
        t->child.push_back(expression());
        match(TokenType::tRP);
        break;
    case TokenType::tNUM: // <factor> ::= NUM
        t->child.push_back(num());
        break;
    default: // wrong procedure call
        t->expr.err=ErrKind::Err;
        syntaxError("unexpected token -> ");
        break;
    }
    ///TODO
    return t;
}
TreeNode* RDParser::relop() {
    TreeNode* t = new TreeNode(NodeKind::Relop, lineno);
    switch (token) {
    case TokenType::tLTE:
        t->oper.oper = OperKind::LTE; break;
    case TokenType::tLT:
        t->oper.oper = OperKind::LT; break;
    case TokenType::tGT:
        t->oper.oper = OperKind::GT; break;
    case TokenType::tGTE:
        t->oper.oper = OperKind::GTE; break;
    case TokenType::tEQ:
        t->oper.oper = OperKind::EQ; break;
    case TokenType::tNEQ:
        t->oper.oper = OperKind::NEQ; break;
    default:
        t->oper.err = ErrKind::Err;
        syntaxError("unexpected token -> ");
        break;
    }
    t->child.push_back(additive_expression());
    return t;
}
TreeNode* RDParser::additive_expression() {
    TreeNode* t = new TreeNode(NodeKind::Expr, lineno);
    
}
TreeNode* RDParser::type_specifier() {
    TreeNode* t = new TreeNode(NodeKind::Type, lineno);
    if (token==TokenType::tINT) {
        t->type.type = TypeKind::Int;
        nextToken();
    }
    else if (token==TokenType::tVOID) {
        t->type.type = TypeKind::Void;
        nextToken();
    }
    else {
        syntaxError("unexpected token -> ");
        t->type.type = TypeKind::Err;
    }
    return t;
}
TreeNode* RDParser::num() {
    TreeNode* t = new TreeNode(NodeKind::Num, lineno);
    if (token==TokenType::tNUM) {
        t->num.str = tokenString;
        nextToken();
    }
    else {
        syntaxError("unexpected token -> ");
    }
    return t;
}