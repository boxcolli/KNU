#ifndef _PARSE_H_
#define _PARSE_H_

#include "globals.h"
#include "bnf.h"

/**************************************************
First Follow
**************************************************/
class FirstFollow {
public:
	FirstFollow(ifstream& fbnf);
private:
	_BNFParser bnfP;
	map<string, set<string>> firsts;
	map<string, set<string>> follows;
	
	void makeFirsts();
	void makeFollows();
};
/**************************************************
Parser
**************************************************/
class LR1Parser {
public:
	LR1Parser();
private:
	_BNFParser bnfP;
};

#endif