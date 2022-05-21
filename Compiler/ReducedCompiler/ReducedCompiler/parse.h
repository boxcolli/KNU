#ifndef _PARSE_H_
#define _PARSE_H_

#include "globals.h"
#include "bnf.h"
#include "state.h"

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
	struct rIndex {
		int rule;	// rule number
		int init;	// at initial item
		set<string> look;	// lookahead
	};
	struct sData {
		set<struct rIndex> k; // kernel item
		set<struct rIndex> c; // closure item
	};
	class LR1State : public FiniteState<sData, string, int, LR1State>  {
	public:
		bool equals(LR1State* s);
	private:
	};

	LR1Parser();


private:
	_BNFParser& bnfP;
};

#endif