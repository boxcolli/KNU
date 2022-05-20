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

    // make grammar g;
    _rule* r = nullptr;
    _rule* r_temp = nullptr;
    pair<TType, string>* s_temp = nullptr;
    while(bnfScanner.process()) {
        string tok = bnfScanner.getToken();
        TType type = bnfScanner.getTtype();
        switch(type) {
        case TType::symbol:
            // write symbol temp
            if (s_temp != nullptr) {
                r->r.push_back(*s_temp);
            }
            // write new symbol temp
            s_temp = new pair<TType, string>(type, tok);
            break;

        case TType::term:
            // write symbol temp
            if (s_temp != nullptr) {
                r->r.push_back(*s_temp);
                s_temp = nullptr;
            }
            // write term
            r->r.push_back(make_pair(type, tok));
            break;

        case TType::assign:
            // write rule
            if (r != nullptr) {
                grammar.push_back(*r);
            }
            // set new rule with symbol temp
            r = new _rule(s_temp->second);
            delete s_temp;
            s_temp = nullptr;
            break;
            
        case TType::op_or:
            // write symbol temp
            if (s_temp != nullptr) {
                r->r.push_back(*s_temp);
                s_temp = nullptr;
            }
            // write rule
            r_temp = new _rule(r->l);
            grammar.push_back(*r);
            // set new rule
            r = r_temp;
            r_temp = nullptr;
            break;

        case TType::error:
        default:
            MSG_EXIT("bToG : error type");
        }
    }
    if (s_temp != nullptr) {
        r->r.push_back(*s_temp);        
    }
    grammar.push_back(*r);


    // make symbol map
    string str = g[0].l;
    int begin = 0;
    int i = 1;
    for ( ; i < g.size(); i++) {
        if (g[i].l != str) {
            symmap[str] = make_pair(begin, i);
            str = g[i].l;
            begin = i;
        }
    }
    symmap[str] = make_pair(begin, i);

    // make term list
    for (auto rule : g) {
        for (auto p : rule.r) {
            if (p.first == TType::term) {
                termset.insert(p.second);
            }
        }
    }

    // make nullable list
    for (auto rule : g) {
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