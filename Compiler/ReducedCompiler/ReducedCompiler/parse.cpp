#include "globals.h"
#include "state.h"
#include "parse.h"
#include "fhead.h"

/**************************************************
First Follow
**************************************************/
void kahn(map<string, set<string>> &data, map<string, set<string>> &adj, map<string, int> &ind) {
    // kahn algorithm
    //      from independent symbol (in-degree==0)
    //      to dependent symbol
    while (true) {
        // find 0 in-degree
        string zero = "";
        for (auto keyval : ind) {
            if (keyval.second==0) {
                zero = keyval.first;
                break;
            }
        }
        if (zero=="") { break; }    // done

        ind.erase(zero);

        // add firsts
        for (auto lhand : adj[zero]) {
            ind[lhand] -= 1; // in-degree--
            for (auto d : data[zero]) {
                data[lhand].insert(d);
            }
        }
    }
}
void kahn_w_cycle(map<string, set<string>> &data, map<string, set<string>> &adj, map<string, int> &ind) {
    // simplify graph
    kahn(data, adj, ind);

    // init
    map<string, set<string>> bind;
    for (auto keyval : ind) {
        bind[keyval.first] = set<string>();
    }

    
}

FirstFollow::FirstFollow(ifstream& fbnf) : bnfP(_BNFParser(fbnf)) {
    makeFirsts();
    makeFollows();
}
void FirstFollow::makeFirsts() {
    // dependency graph : lhand <- rhand-symbol
    map<string, set<string>> adj;
    // in-degree count
    map<string, int> ind;
    // init
    for (auto keyVal : bnfP.symmap) {
        firsts[keyVal.first] = set<string>();
        adj[keyVal.first] = set<string>();
        ind[keyVal.first] = 0;
    }
    // construct graph
    for (auto rule : bnfP.grammar) {
        for (auto rhand : rule.r){
            // term
            if (rhand.first==TType::term) {
                // epsilon rule?
                if (rhand.second==EPSILON) {
                    // do nothing
                }
                else {
                    firsts[rule.l].insert(rhand.second);  // add firsts
                }
            }
            // symbol
            else {
                if (rhand.second != rule.l  // ignore recursion
                    && adj[rhand.second].find(rule.l) == adj[rhand.second].end()) {
                    adj[rhand.second].insert(rule.l);   // graph : lhand <- symbol
                    ind[rule.l] += 1;                       // in-degree for lhand
                }                
                if (bnfP.isNullable(rule.l)) {      // nullable?
                    continue;
                }
            }
            break;
        }
    }

    // kahn algorithm : add firsts
    //      from independent symbol (in-degree==0)
    //      to dependent symbol
    while (true) {
        // find 0 in-degree
        string zero = "";
        for (auto keyval : ind) {
            if (keyval.second==0) {
                zero = keyval.first;
                break;
            }
        }
        if (zero=="") { break; }    // done

        ind.erase(zero);

        // add firsts
        for (auto lhand : adj[zero]) {
            ind[lhand] -= 1; // in-degree--
            for (auto first : firsts[zero]) {
                firsts[lhand].insert(first);
            }
        }
    }
}

void FirstFollow::makeFollows() {
    // dependency graph : lhand <- rhand-symbol
    map<string, set<string>> adj;
    // in-degree count
    map<string, int> ind;
    // init
    for (auto keyVal : bnfP.symmap) {
        follows[keyVal.first] = set<string>();
        adj[keyVal.first] = set<string>();
        ind[keyVal.first] = 0;
    }
    follows[START_SYM].insert(END_SYM);
    // construct graph
    for (auto rule : bnfP.grammar) {
        // right-hand
        for (auto it0 = rule.r.begin();
            it0 != rule.r.end()-1; it0++) {
            if (it0->first==TType::symbol) {
                // r[i] := symbol
                for (auto it1 = it0+1; it1!=rule.r.end();it1++) {
                    if (it1->first==TType::symbol) {
                        // r[i+1] := symbol
                        for (auto f : firsts[it1->second])
                            { follows[it0->second].insert(f); }
                        if (bnfP.isNullable(it1->second))
                            { continue; }
                    }
                    else {
                        // r[i+1] := term
                        follows[it0->second].insert(it1->second);
                    }
                    break;
                }
            }
        }

        // follow(lhand) -> follow(last-symbol)
        for (auto it = rule.r.rbegin(); it != rule.r.rend(); it++) {
            if (it->first==TType::symbol) {
                // only if symbol
                if (rule.l != it->second    // ignore recursion
                    && adj[rule.l].find(it->second) == adj[rule.l].end()) {                    
                    adj[rule.l].insert(it->second);
                    ind[it->second] += 1;
                }                
                if (bnfP.isNullable(it->second))
                    { continue; }
            }
            break;
        }
    }

    cout << "follow_adj:" << endl;
    for (auto keyval : adj) {
        cout << keyval.first << ":";
        for (auto v : keyval.second) {
            cout << " " << v;
        }
        cout << endl;
    }
    cout << endl;

    cout << "follow_ind:" << endl;
    for (auto keyval : ind) {
	    cout << keyval.first << " : " << keyval.second << endl;
    }
    cout << endl;


    // kahn algorithm : add follows
    //      from independent symbol (in-degree==0)
    //      to dependent symbol
    while (true) {
        // find 0 in-degree
        string zero = "";
        for (auto keyValue : ind) {
            if (keyValue.second==0) {
                zero = keyValue.first;
                break;
            }
        }
        if (zero=="") { break; }    // done

        ind.erase(zero);
        cout << "follow..." << zero << endl;

        // add firsts
        for (auto lhand : adj[zero]) {
            ind[lhand] -= 1; // in-degree--
            for (auto f : follows[zero]) {
                follows[lhand].insert(f);
            }
        }
    }
}
/**************************************************
Parser
**************************************************/
bool LR1Parser::LR1State::equals(LR1State* s) {
    return data.k == s->data.k;
}
