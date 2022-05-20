#include "globals.h"
#include "state.h"
#include "parse.h"
#include "fhead.h"

/**************************************************
First Follow
**************************************************/


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
        for (auto tok : rule.r){
            // term
            if (tok.first==TType::term) {
                // epsilon rule?
                if (tok.second==EPSILON) {
                    // do nothing
                }
                else {
                    firsts[rule.l].insert(tok.second);  // add firsts
                }
            }
            // symbol
            else {               
                
                adj[tok.second].insert(rule.l);     // graph : lhand <- symbol
                ind[rule.l] += 1;                   // in-degree for lhand
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
        for (auto keyValue : ind) {
            if (keyValue.second==0) {
                zero = keyValue.first;
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
}
/**************************************************
Parser
**************************************************/
