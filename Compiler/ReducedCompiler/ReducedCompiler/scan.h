#ifndef _SCAN_H_
#define _SCAN_H_

#include "globals.h"
#include "state.h"
#include "fhead.h"

/**************************************************
Basic form of state
**************************************************/
enum class TransitionOpt;

class SingleState : public FiniteState<int, char, TransitionOpt, SingleState> {
public:
	SingleState(int data = -1) : FiniteState(data) {}

	void addMap(int in, SingleState* next, TransitionOpt opt) {
		FiniteState::addMap(in, next, opt);
	}

	pair<SingleState*, TransitionOpt> pushInput(char in) {
		return FiniteState::pushInput(in);
	}
};

/**************************************************
Scanner output type
**************************************************/
enum class TokenType {
	tERROR = -1,
	tNULL,
	tELSE, tIF, tINT, tRETURN, tVOID, tWHILE,
	tID, tNUM,
	tADD, tSUB, tMUL, tDIV, tLT, tLTE, tGT, tGTE, tEQ, tNEQ,
	tASSIGN, tSEMI, tCOMMA,
	tLP, tRP, tLSB, tRSB, tLCB, tRCB,
	tCOMMENT
};

enum class TokErrType {
	eNOERROR,
	eNOMATCHINGTTYPE,
	eINVALIDINPUT,
	eINVALIDRULE,
	eNOCOMMENTEND,
	eENDOFFILE
};

/**************************************************
Scanner
**************************************************/
class Scanner {
public:
	Scanner(ifstream& f);

	TokenType processChar();
	TokenType processToken();

	string getToken() { return tokenBuffer; }
	TokErrType getErrorType() { return errorType; }
	bool isNewLine() { return newline; }

private:
	FileHeader fileHeader;
	map<string, SingleState*> states;	// state list
	SingleState* currentState;	// state cursor
	string tokenBuffer;			// token buffer
	TokErrType errorType;		// recent error
	bool flushFlag;				// flush token buffer on next process
	bool newline;				// recent newline
	
	string EOFSTR = "";
	enum class StateData {
		nonf = 0,
		fID, fNUM,
		fADD, fSUB, fMUL, fDIV,
		fLT, fLTE, fGT, fGTE, fEQ, fNEQ,
		fASSIGN, fSEMI, fCOMMA,
		fLP, fRP, fLSB, fRSB, fLCB, fRCB,
		fCOMMENT
	};

	void addState(string name, StateData data = StateData::nonf) {
		states[name] = new SingleState(static_cast<int>(data));
	}

	void mapState(string from, string to, string instring,
		TransitionOpt opt);

	void buildState();
	void buildInit();
	void buildID();
	void buildNUM();
	void buildDIV_COMMENT();	// processing: /, /**/
	void buildLT_LTE();			// processing: <, <=
	void buildGT_GTE();			// processing: >, >=
	void buildEQ_ASSIGN();		// processing: ==, =
	void buildNEQ();

	TokenType findKeyword(string token);
	TokenType stateToToken(StateData stateData);
};

#endif