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
	_BNFParser bnfP;

	FirstFollow(ifstream& fbnf);
	map<string, set<string>> firsts;
	map<string, set<string>> follows;
	set<string> calcFirst(vector<pair<TType, string>> symterm);

private:
	void makeFirsts();
	void makeFollows();
};
/**************************************************
Parser
**************************************************/
class LR1Parser {
public:
	struct _item {
		int rule;	// rule number
		int init;	// at initial item
		set<string> look;	// lookahead
		bool operator == (const _item& r) const {
			return this->rule == r.rule
				&& this->init == r.init;
		}
		struct _item(int r) : rule(r) {}
		struct _item(int r, int i) : rule(r), init(i) {}
	};
	struct _sdata {
		set<_item> iset;
		bool operator == (const _sdata& d) const {
			return this->iset == d.iset;
		}		
	};
	class LR1State : public FiniteState<_sdata, string, int, LR1State>  {
		bool operator == (const LR1State& s) const {
			return this->data == s.data;
		}
	};

	LR1Parser(ifstream& fbnf);


private:
	FirstFollow&	ff;
	_BNFParser&		bnfP;
	map<string, pair<int, int>>& symmap;
	set<string>& 	termset;	
	set<LR1State>	states;
	_grammar&		grammar;

	pair<TType, string> itemReference(int rule, int init);
	void insertItem(LR1State to, int rule, int init, string lk);
};

#endif