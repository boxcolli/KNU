#ifndef _SCAN_H_
#define _SCAN_H_

#include "globals.h"

const string ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const string NUMDIGIT = "0123456789";
const string OTHERCHAR = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const string WHITESPACE = " \t\n\v\f\r";

/// Returns a copy of the base string after erasing some characters.
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

/**************************************************
Basic form of state
**************************************************/
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

/**************************************************
File read helper	
**************************************************/
class FileHeader {
private:
	ifstream* fin;
	string buffer;
	size_t length;
	int cursor;

	char putBackChar = '\0';
	bool putBackFlag = false;

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

		// buffer empty?
		if (cursor == length) {
			buffer = "";
			getline(*fin, buffer);

			// EOF?
			if (buffer == "" && fin->eof()) {
				return EOF;
			}

			length = buffer.length();
			cursor = 0;
		}

		return buffer[cursor++];
	}

	void putBack(char c) {
		putBackChar = c;
		putBackFlag = true;
	}
};

/**************************************************
Scanner output type
**************************************************/
typedef enum class TokenType {
		tERROR = -1,
		tNULL,
		tELSE, tIF, tINT, tRETURN, tVOID, tWHILE,
		tID, tNUM,
		tADD, tSUB, tMUL, tDIV, tLT, tLTE, tGT, tGTE, tEQ, tNEQ,
		tASSIGN, tENDS, tCOMMA,
		tLP, tRP, tLSB, tRSB, tLCB, tRCB,
		tCOMMENT
	} TokenType;

typedef enum class TokErrType {
	eNOERROR,
	eNOMATCHINGTTYPE,
	eINVALIDINPUT,
	eINVALIDRULE,
	eNOCOMMENTEND,
	eENDOFFILE
} TokErrType;

/******************************
*Scanner	
******************************/
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
	TokErrType errorType;

	// recent newline
	bool newline;

	typedef enum class StateData {
		nonf = 0,
		fID, fNUM,
		fADD, fSUB, fMUL, fDIV,
		fLT, fLTE, fGT, fGTE, fEQ, fNEQ,
		fASSIGN, fENDS, fCOMMA,
		fLP, fRP, fLSB, fRSB, fLCB, fRCB,
		fCOMMENT
	} StateData;

	string EOFSTR = "";

	typedef enum class TransitionOption {
		optNORMAL, optLOOKAHEAD, optDISCARD
	} TransitionOption;


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
			NUMDIGIT + OTHERCHAR + WHITESPACE + EOFSTR,
			TransitionOption::optLOOKAHEAD);
	}
	void buildNUM() {
		mapState("NUM", "NUM", NUMDIGIT);
		addState("fNUM", StateData::fNUM);
		mapState("NUM", "fNUM",
			ALPHABET + OTHERCHAR + WHITESPACE + EOFSTR,
			TransitionOption::optLOOKAHEAD);
	}
	// processing: /, /**/
	void buildDIV_COMMENT() {
		addState("fDIV", StateData::fDIV);
		mapState("DIV_COMMENT", "fDIV",
			ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "*")
			+ WHITESPACE + EOFSTR,
			TransitionOption::optLOOKAHEAD);

		addState("COMMENT");
		mapState("DIV_COMMENT", "COMMENT", "*");
		mapState("COMMENT", "COMMENT",
			ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "*")
			+ WHITESPACE,
			TransitionOption::optDISCARD);	// ignore comments

		addState("COMMENT2");
		mapState("COMMENT", "COMMENT2", "*");
		mapState("COMMENT2", "COMMENT",
			ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "*")
			+ WHITESPACE,
			TransitionOption::optDISCARD);

		addState("COMMENTend", StateData::fCOMMENT);
		mapState("COMMENT2", "COMMENTend", "/",
			TransitionOption::optDISCARD);
	}
	// processing: <, <=
	void buildLT_LTE() {
		addState("fLT", StateData::fLT);
		mapState("LT_LTE", "fLT",
			ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "=")
			+ WHITESPACE + EOFSTR,
			TransitionOption::optLOOKAHEAD);

		addState("fLTE", StateData::fLTE);
		mapState("LT_LTE", "fLTE", "=");
	}
	// processing: >, >=
	void buildGT_GTE() {
		addState("fGT", StateData::fGT);
		mapState("GT_GTE", "fGT",
			ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "=")
			+ WHITESPACE + EOFSTR,
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
			ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "=")
			+ WHITESPACE + EOFSTR,
			TransitionOption::optLOOKAHEAD);
	}
	void buildNEQ() {
		addState("fNEQ", StateData::fNEQ);
		mapState("NEQ", "fNEQ", "=");
	}

	TokenType findKeyword(string token) {
		if (token == "else") return TokenType::tELSE;
		else if (token == "if") return TokenType::tIF;
		else if (token == "int") return TokenType::tINT;
		else if (token == "return") return TokenType::tRETURN;
		else if (token == "void") return TokenType::tVOID;
		else if (token == "while") return TokenType::tWHILE;
		else return TokenType::tID;
	}

	TokenType stateToToken(StateData stateData) {
		switch (stateData) {
		case StateData::fID:		return findKeyword(tokenBuffer);
		case StateData::fNUM:		return TokenType::tNUM;
		case StateData::fADD:		return TokenType::tADD;
		case StateData::fSUB:		return TokenType::tSUB;
		case StateData::fMUL:		return TokenType::tMUL;
		case StateData::fDIV:		return TokenType::tDIV;
		case StateData::fLT:		return TokenType::tLT;
		case StateData::fLTE:		return TokenType::tLTE;
		case StateData::fGT:		return TokenType::tGT;
		case StateData::fGTE:		return TokenType::tGTE;
		case StateData::fEQ:		return TokenType::tEQ;
		case StateData::fNEQ:		return TokenType::tNEQ;
		case StateData::fASSIGN:	return TokenType::tASSIGN;
		case StateData::fENDS:		return TokenType::tENDS;
		case StateData::fCOMMA:		return TokenType::tCOMMA;
		case StateData::fLP:		return TokenType::tLP;
		case StateData::fRP:		return TokenType::tRP;
		case StateData::fLSB:		return TokenType::tLSB;
		case StateData::fRSB:		return TokenType::tRSB;
		case StateData::fLCB:		return TokenType::tLCB;
		case StateData::fRCB:		return TokenType::tRCB;
		default:
			errorType = TokErrType::eNOMATCHINGTTYPE;
			return TokenType::tERROR;
		}
	}

public:

	Scanner(ifstream* f) : fileHeader(FileHeader(f)),
							tokenBuffer(""),
							flushFlag(false),
							errorType(TokErrType::eNOERROR) {
		EOFSTR.push_back(EOF);

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
		
		// newline?
		newline = (in == '\n') ? true : false;

		// flush token?
		if (flushFlag) {
			tokenBuffer.clear();
			flushFlag = false;
		}

		// pushChar to current state
		pair<SingleState*, int> temp = currentState->pushChar(in);
		SingleState* nextState = temp.first;
		int opt = temp.second;

		// no result?
		if (nextState == nullptr) {

			// character
			if (currentState == states["init"]) {
				if (in == EOF) {
					errorType = TokErrType::eENDOFFILE;
				}
				else {
					// set error
					errorType = TokErrType::eINVALIDINPUT;

					// retriev char
					tokenBuffer.push_back(in);

					// flush next time
					flushFlag = true;
				}
			}
			// comment not closed
			else if (currentState == states["COMMENT"]
				|| currentState == states["COMMENTS2"]) {
				errorType = TokErrType::eNOCOMMENTEND;
			}
			// invalid REX rule
			else {
				// force to reset DFA
				currentState = states["init"];

				// set error
				errorType = TokErrType::eINVALIDRULE;

				// re-process current input
				fileHeader.putBack(in);

				// flush next time
				flushFlag = true;
			}

			return TokenType::tERROR;
		}

		// set no error
		errorType = TokErrType::eNOERROR;



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

		// get state data
		StateData stateData = static_cast<StateData>(nextState->getData());
		if (stateData == StateData::nonf) {
			// token not made
			currentState = nextState;
			return TokenType::tNULL;
		}
		else if (stateData == StateData::fCOMMENT) {
			// comment closed
			currentState = states["init"];
			flushFlag = true;
			return TokenType::tNULL;
		}
		else {
			// token has been made
			flushFlag = true;
			currentState = states["init"];
			return stateToToken(stateData);
		}
	}

	TokenType processToken() {
		TokenType t;
		while ((t = processChar()) != TokenType::tNULL) {
			return t;
		}
	}

	string getToken() {
		return tokenBuffer;
	}

	TokErrType getErrorType() {
		return errorType;
	}

	bool isNewLine() {
		return newline;
	}
};



#endif