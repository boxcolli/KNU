#ifndef _BNF_H_
#define _BNF_H_

#include "globals.h"
#include "state.h"
#include "fhead.h"

/**************************************************
BNF Scanner
**************************************************/
class _BNFState : public FiniteState<int, char, int, _BNFState> {
public:
	_BNFState(int data = -1) : FiniteState(data) {}
	
	void addMap(char in, _BNFState* next, int opt = -1) {
		FiniteState::addMap(in, next, opt);
	}

	pair<_BNFState*, int> pushInput(char in) {
		return FiniteState::pushInput(in);
	}
};

class BNFScanner {
public:
	BNFScanner(ifstream* f);

    bool process();  

    enum class TType {
		eof, empty, newline,
		symbol, term, assign, op_or
	};

    TType getType() { return tokenType; }
	string getToken() { return tokenBuffer; }

private:
	map<string, _BNFState*> states;
	FileHeader	fHeader;
	_BNFState * initState;
	_BNFState * currentState;
	string		tokenBuffer;
	TType		tokenType;
	bool		flushData;

	void addState(string name, TType data = TType::empty) {
		states[name] = new _BNFState(static_cast<int>(data));
	}
	void mapState(string from, string to, string instring) {
        _BNFState* fromState = states[from];
        _BNFState* toState = states[to];
        for (char c : instring) {
            fromState->addMap(c, toState);
        }
    }
    TType pushInput(); // TODO
};

/**************************************************
BNF Parser
**************************************************/

typedef string _symbol;
struct _rhand {
    bool nullable;
    
};

class BNFParser {
public:

private:

};

#endif