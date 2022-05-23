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

    if (ind.size()==0) { return; }  // acyclic

    // cyclic
    bool end = true;
    while (true) {
        for (auto kv : ind) {
            // try update adjacent nodes
            string u = kv.first;
            for (auto v : adj[u]) {
                set<string>& v_set = data[v];
                for (auto u_item : data[u]) {
                    if (v_set.find(u_item)==v_set.end()) {
                        // new update
                        end = false;
                        v_set.insert(u_item);
                    }                    
                }
            }
        }
        if (end) { break; }
        end = true;
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

    kahn_w_cycle(firsts, adj, ind);
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

    kahn_w_cycle(follows, adj, ind);
}

set<string> FirstFollow::calcFirst(vector<pair<TType, string>> symterm) {
    set<string> ret;
    for (auto p : symterm) {
        if (p.first==TType::term) {
            // term
            ret.insert(p.second);
        }
        else {
            // symbol
            for (auto symf : firsts[p.second]) {
                ret.insert(symf);
            }
            if (bnfP.isNullable(p.second)) {
                continue;
            }
        }
        break;
    }
    return ret;
}

/**************************************************
Parser
**************************************************/
pair<TType, string> LR1Parser::itemReference(int rule, int init) {
    return grammar[rule].r[init];
}
void LR1Parser::insertItem(LR1State to, int rule, int init, string lk) {
    auto sdata = to.getData();
    auto find = sdata.iset.find(_item(rule, init));
    if (find == sdata.iset.end()) {
        // add rule, init, lookahead
        _item item(rule, init);
        item.look.insert(lk);
        sdata.iset.insert(item);
    }
    else {        
        if (find->look.find(lk)==find->look.end()) {
            // add lookahead
            _item item = *find;
            sdata.iset.erase(find);
            item.look.insert(lk);
            sdata.iset.insert(item);
        }
        else {
            // cycle, end
            return;
        }
    }    
    // recursion
    if (itemReference(rule, init).first==TType::symbol) {
        int ruleI = symmap[itemReference(rule, init).second].first;
        int ruleEnd = symmap[itemReference(rule, init).second].second;
        for (; ruleI != ruleEnd; ruleI++) {
            grammar[ruleI]
        }
    }
}
LR1Parser::LR1Parser(ifstream& fbnf) :
    ff(FirstFollow(fbnf)), bnfP(ff.bnfP), symmap(bnfP.symmap), termset(bnfP.termset), grammar(bnfP.grammar) {
    
    
    
    _sdata initData;
    for (int rule = symmap[START_SYM].first; rule < symmap[START_SYM].second; rule++) {
        struct _item it(rule, 0);
        if (initData.k.find(it)==initData.k.end()) {

        }
    }
    
    //initData.k.insert()
    LR1State initState();
    
    states.insert(initState);
}