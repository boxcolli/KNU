#ifndef _SCAN_H_
#define _SCAN_H_

#include "globals.h"

const string ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
						abcdefghijklmnopqrstuvwxyz";
const string NUMDIGIT = "0123456789";
const string OTHERCHAR = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const string WHITESPACE = " \t\n\v\f\r";

/// <summary>
/// Returns a copy of the base string after erasing some characters.
/// </summary>
string dropChars(const string base, string chars) {
	string s = base;
	for (auto it = s.begin(); it != s.end(); it++) {
		for (char c : chars) {
			if (*it == c) {
				s.erase(it--);
				break;
			}
		}
	}
	return s;
}

/// Basic form of state for any usage.
class SingleState {
private:

	/// Can hold any optional data. For example, non-final, final or else.
	int data;

	/// Holds transition data, with transition option
	map<char, pair<SingleState*, int>> transition;

public:

	/// Class SingleState constructor
	/// data: int value for optional data
	SingleState(int data = -1) : data(data) {}

	/// Add a transition from this state to another.
	/// in: A single input character
	/// next: Pointer to another state
	void addMap(char in, SingleState* next, int opt) {
		transition[in] = make_pair(next, opt);
	}

	/// Push a single input character to make transition
	/// in: A single input character</param>
	/// returns nullptr if not found or mapped. SingleState* if found.
	pair<SingleState*, int> pushChar(char in) {
		auto search = transition.find(in);
		if (search == transition.end()) {
			return make_pair(nullptr, -1);
		}
		else {
			return search->second;
		}
	}

	/// returns assigned data for this state.
	int getData() {
		return data;
	}
};

class FileHeader {
private:
	ifstream* fin;
	string buffer;
	int length;
	int cursor;

	char putBackChar;
	bool putBackFlag;

public:
	FileHeader(ifstream* fin) : fin(fin) {
		getline(*fin, buffer);
		length = buffer.length();
		cursor = 0;
	}

	char getChar() {
		// have pushed back char
		if (putBackFlag) {
			putBackFlag = false;
			return putBackChar;
		}
		// buffer not empty
		else if (cursor != length) {
			return buffer[cursor++];
		}
		// buffer empty
		else {
			getline(*fin, buffer);
			if (fin->eof()) {
				return EOF;
			}
			length = buffer.length();
			cursor = 0;
			return '\n';
		}
	}

	void putBack(char c) {
		putBackChar = c;
		putBackFlag = true;
	}
};

class Scanner {
private:
	FileHeader fileHeader;

	map<string, SingleState*> states;

	// state cursor
	SingleState* currentState;

	// token buffer
	string tokenBuffer;

	// flush token buffer on next process
	bool flushFlag;

	// recent error
	int errorType;

	

	typedef enum class StateData {
		nonf = 0,
		fID, fNUM,
		fADD, fSUB, fMUL, fDIV,
		fLT, fLTE, fGT, fGTE, fEQ, fNEQ,
		fASSIGN, fENDS, fCOMMA,
		fLP, fRP, fLSB, fRSB, fLCB, fRCB,
		fCOMMENT
	};

	typedef enum class TransitionOption {
		optNORMAL, optLOOKAHEAD, optDISCARD
	};


	void addState(string name, StateData data = StateData::nonf) {
		states[name] = new SingleState(static_cast<int>(data));
	}

	void mapState(string from, string to,
		string instring,
		TransitionOption opt = TransitionOption::optNORMAL) {
		SingleState* s1 = states[from];
		SingleState* s2 = states[to];
		for (char in : instring) {
			s1->addMap(in, s2, static_cast<int>(opt));
		}
	}

	void buildState() {
		// Construct Automata
		addState("init");
		addState("ID");
		addState("NUM");
		addState("ADD", StateData::fADD);
		addState("SUB", StateData::fSUB);
		addState("MUL", StateData::fMUL);
		addState("DIV_COMMENT");
		addState("LT_LTE");
		addState("GT_GTE");
		addState("EQ_ASSIGN");
		addState("NEQ");
		addState("ENDS", StateData::fENDS);
		addState("COMMA", StateData::fCOMMA);
		addState("LP", StateData::fLP);
		addState("RP", StateData::fRP);
		addState("LSB", StateData::fLSB);
		addState("RSB", StateData::fRSB);
		addState("LCB", StateData::fLCB);
		addState("RCB", StateData::fRCB);
	}
	void buildInit() {
		mapState("init", "init", WHITESPACE, TransitionOption::optDISCARD);
		mapState("init", "ID", ALPHABET);
		mapState("init", "NUM", NUMDIGIT);
		mapState("init", "ADD", "+");
		mapState("init", "SUB", "-");
		mapState("init", "MUL", "*");
		mapState("init", "DIV_COMMENT", "/");
		mapState("init", "LT_LTE", "<");
		mapState("init", "GT_GTE", ">");
		mapState("init", "EQ_ASSIGN", "=");
		mapState("init", "NEQ", "!");
		mapState("init", "ENDS", ";");
		mapState("init", "COMMA", ",");
		mapState("init", "LP", "(");
		mapState("init", "RP", ")");
		mapState("init", "LSB", "[");
		mapState("init", "RSB", "]");
		mapState("init", "LCB", "{");
		mapState("init", "RCB", "}");
	}
	void buildID() {
		mapState("ID", "ID", ALPHABET);

		addState("fID", StateData::fID);
		mapState("ID", "fID",
			NUMDIGIT + OTHERCHAR + WHITESPACE,
			TransitionOption::optLOOKAHEAD);
	}
	void buildNUM() {
		mapState("NUM", "NUM", NUMDIGIT);
		addState("fNUM", StateData::fNUM);
		mapState("NUM", "fNUM",
			ALPHABET + OTHERCHAR + WHITESPACE,
			TransitionOption::optLOOKAHEAD);
	}
	// processing: /, /**/
	void buildDIV_COMMENT() {
		addState("fDIV", StateData::fDIV);
		mapState("DIV_COMMENT", "fDIV",
			ALPHABET + NUMDIGIT +
			dropChars(OTHERCHAR, "*")
			+ WHITESPACE,
			TransitionOption::optLOOKAHEAD);

		addState("COMMENT");
		mapState("DIV_COMMENT", "COMMENT", "*");
		mapState("COMMENT", "COMMENT",
					ALPHABET + NUMDIGIT +
					dropChars(OTHERCHAR, "*")
					+ WHITESPACE);	// ignore comments

		addState("COMMENT2");
		mapState("COMMENT", "COMMENT2", "*");
		mapState("COMMENT2", "COMMENT",
					ALPHABET + NUMDIGIT +
					dropChars(OTHERCHAR, "*")
					+ WHITESPACE);

		addState("COMMENTend", StateData::fCOMMENT);
		mapState("COMMENT2", "COMMENTend", "/");
	}
	// processing: <, <=
	void buildLT_LTE() {
		addState("fLT", StateData::fLT);
		mapState("LT_LTE", "fLT",
			ALPHABET + NUMDIGIT +
			dropChars(OTHERCHAR, "=")
			+ WHITESPACE,
			TransitionOption::optLOOKAHEAD);

		addState("fLTE", StateData::fLTE);
		mapState("LT_LTE", "fLTE", "=");
	}
	// processing: >, >=
	void buildGT_GTE() {
		addState("fGT", StateData::fGT);
		mapState("GT_GTE", "fGT",
			ALPHABET + NUMDIGIT +
			dropChars(OTHERCHAR, "=")
			+ WHITESPACE,
			TransitionOption::optLOOKAHEAD);

		addState("fGTE", StateData::fGTE);
		mapState("GT_GTE", "fGTE", "=");
	}
	// processing: ==, =
	void buildEQ_ASSIGN() {
		addState("fEQ", StateData::fEQ);
		mapState("EQ_ASSIGN", "fEQ", "=");

		addState("fASSIGN", StateData::fASSIGN);
		mapState("EQ_ASSIGN", "fASSIGN",
			ALPHABET + NUMDIGIT +
			dropChars(OTHERCHAR, "=")
			+ WHITESPACE,
			TransitionOption::optLOOKAHEAD);
	}
	void buildNEQ() {
		addState("fNEQ", StateData::fNEQ);
		mapState("NEQ", "fNEQ", "=");
	}

	int findKeyword(string token) {
		if (token == "else") return tELSE;
		else if (token == "if") return tIF;
		else if (token == "int") return tINT;
		else if (token == "return") return tRETURN;
		else if (token == "void") return tVOID;
		else if (token == "while") return tWHILE;
		else return tID;
	}

public:

	typedef enum TokenType {
		tERROR = -1,
		tNULL,
		tELSE, tIF, tINT, tRETURN, tVOID, tWHILE,
		tID, tNUM,
		tADD, tSUB, tMUL, tDIV, tLT, tLTE, tGT, tGTE, tEQ, tNEQ,
		tASSIGN, tENDS, tCOMMA,
		tLP, tRP, tLSB, tRSB, tLCB, tRCB,
		tCOMMENT
	};

	typedef enum ErrorType {
		eNOERROR,
		eNOMATCHINGTTYPE,
		eINVALIDINPUT,
		eENDOFFILE
	};

	Scanner(ifstream* f) : fileHeader(FileHeader(f)),
							tokenBuffer(""),
							flushFlag(false),
							errorType(eNOERROR) {
		// build DFA
		buildState();
		buildInit();
		buildID();
		buildNUM();
		buildDIV_COMMENT();
		buildLT_LTE();
		buildGT_GTE();
		buildEQ_ASSIGN();
		buildNEQ();

		currentState = states["init"];
	}

	TokenType processChar() {
		char in = fileHeader.getChar();

		// EOF?
		if (in == EOF) {
			errorType = eENDOFFILE;
			return tERROR;
		}

		// flush token?
		if (flushFlag) {
			tokenBuffer.clear();
			flushFlag = false;
		}

		// retrieve result from current state
		pair<SingleState*, int> temp = currentState->pushChar(in);
		SingleState* nextState = temp.first;
		int opt = temp.second;

		// no result?
		if (nextState == nullptr) {
			// force to reset DFA
			currentState = states["init"];

			// collect char
			tokenBuffer.push_back(in);

			// flush next time
			flushFlag = true;

			// set error
			errorType = eINVALIDINPUT;

			cout << "StateData(" << currentState->getData() << ")\n";

			return tERROR;
		}

		// set no error
		errorType = eNOERROR;

		// process transition option
		switch (static_cast<TransitionOption>(opt)) {
		case TransitionOption::optLOOKAHEAD:
			fileHeader.putBack(in);
			break;
		case TransitionOption::optDISCARD:
			// discard
			break;
		case TransitionOption::optNORMAL:
			tokenBuffer.push_back(in);
			break;
		default:
			tokenBuffer.push_back(in);
		}

		// process state data
		StateData stateData = static_cast<StateData>(nextState->getData());
		if (stateData == StateData::nonf) {
			currentState = nextState;
			return tNULL;
		}
		else {
			flushFlag = true;
			currentState = states["init"];
			switch (stateData) {
			case StateData::fID:		return static_cast<TokenType>(findKeyword(tokenBuffer));
			case StateData::fNUM:		return tNUM;
			case StateData::fADD:		return tADD;
			case StateData::fSUB:		return tSUB;
			case StateData::fMUL:		return tMUL;
			case StateData::fDIV:		return tDIV;
			case StateData::fLT:		return tLT;
			case StateData::fLTE:		return tLTE;
			case StateData::fGT:		return tGT;
			case StateData::fGTE:		return tGTE;
			case StateData::fEQ:		return tEQ;
			case StateData::fNEQ:		return tNEQ;
			case StateData::fASSIGN:	return tASSIGN;
			case StateData::fENDS:		return tENDS;
			case StateData::fCOMMA:		return tCOMMA;
			case StateData::fLP:		return tLP;
			case StateData::fRP:		return tRP;
			case StateData::fLSB:		return tLSB;
			case StateData::fRSB:		return tRSB;
			case StateData::fLCB:		return tLCB;
			case StateData::fRCB:		return tRCB;
			case StateData::fCOMMENT:	return tCOMMENT;
			default:
				errorType = eNOMATCHINGTTYPE;
				return tERROR;
			}
		}
	}

	void processToken() {

	}

	string getToken() {
		return tokenBuffer;
	}

	ErrorType getErrorType() {
		return static_cast<ErrorType>(errorType);
	}
};



#endif