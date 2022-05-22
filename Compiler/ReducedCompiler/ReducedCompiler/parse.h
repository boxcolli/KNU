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
	map<string, set<string>> firsts;
	map<string, set<string>> follows;

private:
	_BNFParser bnfP;
	void kahn(map<string, set<string>> &data, map<string, set<string>> &adj, map<string, int> &ind);
	void kahn_w_cycle(map<string, set<string>> &data, map<string, set<string>> &adj, map<string, int> &ind);
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
		bool operator == (const rIndex& r) const {
			return this->rule == r.rule && this->init == r.init;
		}
	};
	struct sData {
		set<rIndex> k; // kernel item
		set<rIndex> c; // closure item
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