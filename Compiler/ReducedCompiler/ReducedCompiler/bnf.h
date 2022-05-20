#ifndef _BNF_H_
#define _BNF_H_

#include "globals.h"

enum class TType {
	error, symbol, term, assign, op_or
};

struct _rule {
	string l;	// left hand
	vector<pair<TType, string>> r;	// right hand
	
	_rule() {}
	_rule(string left) : l(left) {}
};

typedef vector<_rule> grammar;

/**************************************************
BNF Scanner
**************************************************/
class _BNFScanner {
public:
	_BNFScanner(ifstream& fbnf);
    bool process();
    
	string getToken() { return token; }
	TType getTtype() { return ttype; }

private:
	ifstream& fbnf;
	string buffer;
	vector<string> tokens;
	string token;
	TType ttype;
};

/**************************************************
BNF Parser
**************************************************/
class _BNFParser {
public:
	grammar grammar;
	map<string, pair<int, int>> symmap;
	set<string> termset;
	set<string> nullable;

	_BNFParser(ifstream& f);
	bool isNullable(string symbol);
	grammar::iterator ruleBegin(string lhand);
	grammar::iterator ruleEnd(string lhand);

private:
	/* 3 important output */
	
};

#endif