#include "globals.h"
#include "bnf.h"
#include "state.h"
#include "fhead.h"
/**************************************************
BNF Scanner
**************************************************/
vector<string> tokenize(string src, string del) {
    char* str = new char[src.length()+1];
    strcpy_s(str, src.length()+1, src.c_str());
    char* d = new char[del.length()+1];
    strcpy_s(d, del.length()+1, del.c_str());
    
    char* context = NULL;
    char* token = strtok_s(str, d, &context);
    vector<string> ret;
    while (token != NULL) {
        ret.push_back(token);
        token = strtok_s(NULL, d, &context);
    }
    return ret;
}

_BNFScanner::_BNFScanner(ifstream& fbnf) : 
    fbnf(fbnf), buffer(""), token(""), ttype(TType::error) {};

bool _BNFScanner::process() {
    // get tokens
    if (tokens.size() == 0) {
        while (true) {
            buffer = "";
            getline(fbnf, buffer);
            tokens = tokenize(buffer, " ");
            if (tokens.size() != 0) break;
            if (fbnf.eof())         return false;
        }
    }

    // token = pop_front
    token = tokens[0];
    tokens.erase(tokens.begin());
    
    // assign token type
    if (token[0] == '<' && token[token.length()-1] == '>') {
        ttype = TType::symbol;
    }
    else if (token == "::=") {
        ttype = TType::assign;
    }
    else if (token == "|") {
        ttype = TType::op_or;
    }
    else {
        for (char c : token)
            if (!isalpha(c))
                MSG_EXIT("bnf : invalid rule at..." + buffer);
        ttype = TType::term;
    }

    return true;
}

/**************************************************
BNF Parser
**************************************************/
_BNFParser::_BNFParser(ifstream& fbnf) {
    _BNFScanner bnfScanner(fbnf);

    _rule rule;
    bool wasSym = false;
    string lastSym;
    string tok;
    TType type;
    
    // iterate 0 : symbol
    bnfScanner.process();
    tok = bnfScanner.getToken();
    type = bnfScanner.getTtype();
    if (type!=TType::symbol) { MSG_EXIT("bnf : invalid left-hand"); }
    rule.l = tok;
    // iterate 1 : assign
    bnfScanner.process();
    type = bnfScanner.getTtype();
    if (type != TType::assign) { MSG_EXIT("bnf : expected assign, but not found"); }
    wasSym = false;
    // iterate 2~
    while (bnfScanner.process()) {
        tok = bnfScanner.getToken();
        type = bnfScanner.getTtype();
        switch (type) {
        case TType::symbol:
            if (wasSym) // symbol -> rule.right-hand
                { rule.r.push_back(make_pair(TType::symbol, lastSym)); }
            wasSym = true;  // memo symbol
            lastSym = tok;
            break;

        case TType::term:
            if (wasSym) // symbol -> rule.right-hand
                { rule.r.push_back(make_pair(TType::symbol, lastSym)); }
            wasSym = false;
            rule.r.push_back(make_pair(type, tok)); // memo term
            break;

        case TType::assign:
            if (wasSym) { // rule -> grammar
                grammar.push_back(rule);
                rule = _rule(lastSym);
            }
            else {
                MSG_EXIT("bnf : invalid left-hand");
            }            
            wasSym = false;
            break;

        case TType::op_or:
            if (wasSym) // symbol -> rule.right-hand
                { rule.r.push_back(make_pair(TType::symbol, lastSym)); }
            wasSym = false;
            grammar.push_back(rule);    // rule -> grammar
            rule = _rule(rule.l);             // reuse rule.left-hand
            break;

        case TType::error:
        default:
            MSG_EXIT("bnf : error token type");
        }
    }
    if (wasSym) // symbol -> rule.right-hand
        { rule.r.push_back(make_pair(TType::symbol, lastSym)); }
    grammar.push_back(rule);


    // make symbol map
    string str = grammar[0].l;
    int begin = 0;
    int i = 1;
    for ( ; i < grammar.size(); i++) {
        if (grammar[i].l != str) {
            symmap[str] = make_pair(begin, i);
            str = grammar[i].l;
            begin = i;
        }
    }
    symmap[str] = make_pair(begin, i);


    // make term list
    for (auto rule : grammar) {
        for (auto p : rule.r) {
            if (p.first == TType::term) {
                termset.insert(p.second);
            }
        }
    }


    // make nullable list
    for (auto rule : grammar) {
        if (rule.r[0].second == EPSILON) {
            nullable.insert(rule.l);
        }
    }
}

bool _BNFParser::isNullable(string symbol) {
    return nullable.find(symbol) != nullable.end();
}
grammar::iterator _BNFParser::ruleBegin(string lhand) {
    return grammar.begin() + symmap[lhand].first;
}
grammar::iterator _BNFParser::ruleEnd(string lhand) {
    return grammar.begin() + symmap[lhand].second;
}