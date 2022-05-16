#ifndef _PARSE_H_
#define _PARSE_H_

#include "globals.h"
#include "state.h"
#include "fhead.h"

typedef struct _rule {
	string L;
	vector<string> R;
};
typedef vector<_rule> _grammar;

/**************************************************
Grammar Handler
**************************************************/
class BNFState : public FiniteState<int, char, int, BNFState> {
public:
	BNFState(int data = -1) : FiniteState(data) {}
	void addMap(int in, BNFState* next, int opt = -1) {
		FiniteState::addMap(in, next, opt);
	}
	pair<BNFState*, int> pushInput(char in) {
		return FiniteState::pushInput(in);
	}
};

class GrammarHandler {
public:
	GrammarHandler(ifstream* f);
private:
	FileHeader fHeader;
	map<string, BNFState*> states;
	BNFState* currentState;
	string tokenBuffer;

	enum class StateData {
		nonf, f
	};

	void addState(string name, StateData data = StateData::nonf) {
		states[name] = new BNFState(static_cast<int>(data));
	}
	void mapState(string from, string to, string instring);
	
	void buildStates();
};
/**************************************************
First Follow
**************************************************/
class FirstFollow {
public:
	FirstFollow();
private:
	// map term
	// map nont

	// map 

	// map first
	// map follow
	
};
/**************************************************
Parser
**************************************************/
class LR1Parser {
public:
	LR1Parser();
private:
	
};

#endif