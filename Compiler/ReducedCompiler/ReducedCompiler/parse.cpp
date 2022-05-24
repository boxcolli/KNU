#include "globals.h"
#include "scan.h"
#include "tree.h"
#include "parse.h"


/**************************************************
Parser
**************************************************/

RDParser::RDParser(ifstream& fcode) :
    fcode(fcode), scanner(Scanner(fcode)),
    root(nullptr), token(TokenType::tNULL),
    tokenString(""), lineno(0), error(false) {
    
    nextToken();
    root = declaration_list();
    if (token!=TokenType::tERROR) {
        syntaxError("Code ends before file\n");
    }
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
    TreeNode* q;
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
        q = declaration();
        p->sibling = q;
        p = q;
    }
    return t;
}

TreeNode* RDParser::declaration() {
    TreeNode* t = new TreeNode(NodeKind::Decl, lineno);
    // <type-specifier>    
    t->child.push_back(type_specifier());
    // ID
    if (token==TokenType::tID) {
        t->decl.id = new string(tokenString);        
        nextToken();
    }
    else { syntaxError("unexpected token -> "); }
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
    TreeNode* p = nullptr;
    TreeNode* q = nullptr;
    if (token==TokenType::tVOID) {
        // VOID
        t->paraml.empty = true;
    }
    else {
        // <param> { COMMA <param> }
        // <param> ::= <type-specifier> ID [ LSB RSB ]
        p = new TreeNode(NodeKind::Param, lineno);
        p->child.push_back(type_specifier());
        if (token==TokenType::tID) {
            p->param.id = new string(tokenString);
            nextToken();
        }
        else { syntaxError("unexpected token -> "); }
        if (token==TokenType::tLSB) {
            p->param.ary = true;
            match(TokenType::tLSB);
            match(TokenType::tRSB);
        }
        t->child.push_back(p);
        while (token==TokenType::tCOMMA) {
            match(TokenType::tCOMMA);
            q = new TreeNode(NodeKind::Param, lineno);
            q->child.push_back(type_specifier());
            if (token==TokenType::tID) {
                q->param.id = new string(tokenString);
                nextToken();
            }
            else { syntaxError("unexpected token -> "); }
            if (token==TokenType::tLSB) {
                q->param.ary = true;
                match(TokenType::tLSB);            
                match(TokenType::tRSB);
            }            
            p->sibling=q;
            p=q;
        }
    }
    return t;
}

TreeNode* RDParser::local_declarations() {
    TreeNode* t = new TreeNode(NodeKind::LocDecl, lineno);
    TreeNode* p = nullptr;
    TreeNode* q = nullptr;
    
    if ((token==TokenType::tINT)
        ||(token==TokenType::tVOID)) {        
        // <type-specifier>
        p = new TreeNode(NodeKind::VarDecl, lineno);
        p->child.push_back(type_specifier());
        // ID
        if (token==TokenType::tID) {
            p->decl.id = new string(tokenString);
            nextToken();        
        } else { syntaxError("unexpected token -> "); }
        // SEMI | LSB
        if (token==TokenType::tLSB) {
            t->decl.ary=true;
            match(TokenType::tLSB);
            t->child.push_back(num());
            match(TokenType::tRSB);
        }
        match(TokenType::tSEMI);
        t->child.push_back(p);
        while ((token==TokenType::tINT)
            || (token==TokenType::tVOID)) {
            q = new TreeNode(NodeKind::VarDecl, lineno);
            // <type-specifier>
            q->child.push_back(type_specifier());
            // ID
            if (token==TokenType::tID) {
                q->decl.id = new string(tokenString);
                nextToken();
            } else { syntaxError("unexpected token -> "); }
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
    TreeNode* p = nullptr;
    TreeNode* q = nullptr;
    if ((token==TokenType::tID)
        ||(token==TokenType::tIF)
        ||(token==TokenType::tLCB)
        ||(token==TokenType::tLP)
        ||(token==TokenType::tNUM)
        ||(token==TokenType::tRETURN)
        ||(token==TokenType::tSEMI)
        ||(token==TokenType::tWHILE)) { // firsts of <statement-list>
        p = statement();
        t->child.push_back(p);
        while ((token==TokenType::tID)
            ||(token==TokenType::tIF)
            ||(token==TokenType::tLCB)
            ||(token==TokenType::tLP)
            ||(token==TokenType::tNUM)
            ||(token==TokenType::tRETURN)
            ||(token==TokenType::tSEMI)
            ||(token==TokenType::tWHILE)) {
            q = statement();
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
        t = new TreeNode(NodeKind::ERROR, lineno); break;
    }
    return t;
}

TreeNode* RDParser::expression_stmt() {
    TreeNode* t = new TreeNode(NodeKind::ExprStmt, lineno);
    if (token==TokenType::tSEMI) {        
        t->exps.empty=true;
    }
    else { t->child.push_back(expression()); }
    match(TokenType::tSEMI);
    return t;
}

TreeNode* RDParser::compound_stmt() {
    TreeNode* t = new TreeNode(NodeKind::CmpdStmt, lineno);
    match(TokenType::tLCB);
    t->child.push_back(local_declarations());
    t->child.push_back(statement_list());
    match(TokenType::tRCB);
    return t;
}

TreeNode* RDParser::selection_stmt() {
    TreeNode* t = new TreeNode(NodeKind::SlctStmt, lineno);
    match(TokenType::tIF);
    match(TokenType::tLP);
    t->child.push_back(expression());
    match(TokenType::tRP);
    t->child.push_back(statement());
    if (token==TokenType::tELSE) {
        t->slct.els = true;
        match(TokenType::tELSE);
        t->child.push_back(statement());        
    }
    return t;
}

TreeNode* RDParser::iteration_stmt() {
    TreeNode* t = new TreeNode(NodeKind::IterStmt, lineno);
    match(TokenType::tWHILE);
    match(TokenType::tLP);
    t->child.push_back(expression());
    match(TokenType::tRP);
    t->child.push_back(statement());
    return t;
}

TreeNode* RDParser::return_stmt() {
    TreeNode* t = new TreeNode(NodeKind::RetStmt, lineno);
    match(TokenType::tRETURN);
    if(token==TokenType::tSEMI) {
        t->ret.empty = true;
    }
    else { t->child.push_back(expression());  }
    match(TokenType::tSEMI);
    return t;
}

TreeNode* RDParser::expression() {    
    TreeNode* t = new TreeNode(NodeKind::Expr, lineno);
    TreeNode* p = factor();
    TreeNode* q = nullptr;
    switch (p->nodeKind) {
    case NodeKind::Expr: // LP <expression> RP
        break;
    case NodeKind::Var: // operation        
        if (token==TokenType::tASSIGN) {
            match(TokenType::tASSIGN);
            q = new TreeNode(NodeKind::Assign, lineno);
            q->child.push_back(p);
            q->child.push_back(expression());
            t->child.push_back(q);
            break;
        }            
        // else: continue
    case NodeKind::Num: // [ <relop> <additive-expression> ]
        if ((token==TokenType::tLTE)
            ||(token==TokenType::tLT)
            ||(token==TokenType::tGT)
            ||(token==TokenType::tGTE)
            ||(token==TokenType::tEQ)
            ||(token==TokenType::tNEQ)
            ||(token==TokenType::tADD)
            ||(token==TokenType::tSUB)
            ||(token==TokenType::tMUL)
            ||(token==TokenType::tDIV)) {
            q = new TreeNode(NodeKind::Oper, lineno);
            q->child.push_back(p);
            switch(token) {
            case TokenType::tLTE:                
                q->oper.oper = OperKind::LTE;
                nextToken();
                q->child.push_back(additive_expression());
                break;
            case TokenType::tLT:
                q->oper.oper = OperKind::LT;
                nextToken();
                q->child.push_back(additive_expression());
                break;
            case TokenType::tGT:
                q->oper.oper = OperKind::GT;
                nextToken();
                q->child.push_back(additive_expression());
                break;
            case TokenType::tGTE:
                q->oper.oper = OperKind::GTE;
                nextToken();
                q->child.push_back(additive_expression());
                break;
            case TokenType::tEQ:
                q->oper.oper = OperKind::EQ;
                nextToken();
                q->child.push_back(additive_expression());
                break;
            case TokenType::tNEQ:
                q->oper.oper = OperKind::NEQ;
                nextToken();
                q->child.push_back(additive_expression());
                break;
            case TokenType::tADD:
                q->oper.oper = OperKind::ADD;
                nextToken();
                q->child.push_back(term());
                break;
            case TokenType::tSUB:
                q->oper.oper = OperKind::SUB;
                nextToken();
                q->child.push_back(term());
                break;
            case TokenType::tMUL:
                q->oper.oper = OperKind::MUL;
                nextToken();
                q->child.push_back(factor());
                break;
            case TokenType::tDIV:
            default:
                q->oper.oper = OperKind::DIV;
                nextToken();
                q->child.push_back(factor());
                break;
            }            
            t->child.push_back(q);
        }
        else { // no operation, only Var | Num
            t->child.push_back(p);
        }
        break;
    }
    return t;
}
TreeNode* RDParser::additive_expression() {
    // <term> { <addop> <term> }
    TreeNode* t = term();
    TreeNode* p = t; // previous <addop>
    TreeNode* q = nullptr; // new <addop>
    TreeNode* r = nullptr; // previous <term>
    bool firstOper = true;
    while((token==TokenType::tADD)
        ||(token==TokenType::tSUB)) {        
        q = new TreeNode(NodeKind::Addop, lineno);
        if(token==TokenType::tADD) { 
            q->oper.oper = OperKind::ADD; }
        else {
            q->oper.oper = OperKind::SUB; }
        nextToken();
        if (firstOper) {
            firstOper = false;
            q->child.push_back(t);
            t = p = q;
            r = term(); /*
                Always, this last term
                will be added only after
                there is no more Addop. */
        } else {
            q->child.push_back(r);
            p->child.push_back(q);
            p = q;
            r = term();
        }
    }
    if (r!=nullptr) { p->child.push_back(r); }
    return t;
}

TreeNode* RDParser::term() {
    // <factor> { <mulop> <factor> }
    TreeNode* t = factor();
    TreeNode* p = t; // same structure as : additive_expression()
    TreeNode* q = nullptr;
    TreeNode* r = nullptr;
    bool firstOper = true;
    while((token==TokenType::tMUL)
        ||(token==TokenType::tDIV)) {
        q = new TreeNode(NodeKind::Mulop, lineno);                
        if(token==TokenType::tMUL) {
            q->oper.oper = OperKind::MUL;
        } else {
            q->oper.oper = OperKind::DIV; }
        nextToken();
        if(firstOper) {
            firstOper = false;
            q->child.push_back(t);
            t = p = q;
            r = factor();            
        } else {
            q->child.push_back(r);
            p->child.push_back(q);
            p = q;
            r = factor();
        }
    }
    if (r!=nullptr) { p->child.push_back(r); }
    return t;    
}

TreeNode* RDParser::factor() {
    TreeNode* t = nullptr;
    TreeNode* p = nullptr;
    string id;
    switch (token) {
    // LP <expression> RP
    case TokenType::tLP:
        match(TokenType::tLP);
        t = expression();
        match(TokenType::tRP);
        break;
    // <var> | <call>
    case TokenType::tID:
        id = tokenString;
        nextToken();
        switch (token) {
        case TokenType::tLSB: // <var> ::= ID [ LSB <expression> RSB ]
            t = new TreeNode(NodeKind::Var, lineno);
            t->var.id= new string(id);
            match(TokenType::tLSB);
            t->child.push_back(expression());
            match(TokenType::tRSB);
            break;
        case TokenType::tLP: // <call> ::= ID LP <args> RP
            t = new TreeNode(NodeKind::Call, lineno);
            t->call.id = new string(id);
            match(TokenType::tLP);
            t->child.push_back(args());
            match(TokenType::tRP);
            break;
        default: // <var> ::= ID
            t = new TreeNode(NodeKind::Var, lineno);
            t->var.id = new string(id);
            break;
        }
        break;
    case TokenType::tNUM: // <factor> ::= NUM
        t = num();
        break;
    default:
        syntaxError("unexpected token -> ");
        t = new TreeNode(NodeKind::ERROR, lineno);
    }
    return t;
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

TreeNode* RDParser::args() {
    TreeNode* t = new TreeNode(NodeKind::Args, lineno);
    TreeNode* p = t;
    if ((token==TokenType::tID)
        ||(token==TokenType::tLP)
        ||(token==TokenType::tNUM)) {
        p = expression();
        t->child.push_back(p);
        while(token==TokenType::tCOMMA) {
            match(TokenType::tCOMMA);
            p->sibling=expression();
            p=p->sibling;
        }
    }
    else { t->args.empty = true; }
    return t;
}

TreeNode* RDParser::num() {
    TreeNode* t = new TreeNode(NodeKind::Num, lineno);
    if (token==TokenType::tNUM) {
        t->num.str = new string(tokenString);
        nextToken();
    }
    else {
        syntaxError("unexpected token -> ");
    }
    return t;
}