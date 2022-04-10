#include <fstream>
#include <map>
#include <string>

using namespace std;

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

/// <summary>
/// Basic form of state for any usage.
/// </summary>
class SingleState {
private:

	/// <summary>
	/// Can hold any optional data. For example, non-final, final or else.
	/// </summary>
	int data;

	/// <summary>
	/// Holds transition data, with transition option
	/// </summary>
	map<char, pair<SingleState*, int>> transition;

public:

	/// <summary>
	/// Class SingleState constructor
	/// </summary>
	/// <param name="data">int value for optional data</param>
	SingleState(int data=-1) : data(data) {}

	/// <summary>
	/// Add a transition from this state to another.
	/// </summary>
	/// <param name="in">A single input character</param>
	/// <param name="next">Pointer to another state</param>
	void addMap(char in, SingleState* next, int opt) {
		transition[in] = make_pair(next, opt);
	}

	/// <summary>
	/// Push a single input character to make transition
	/// </summary>
	/// <param name="in">A single input character</param>
	/// <returns>nullptr if not found or mapped. SingleState* if found.</returns>
	pair<SingleState*, int> pushChar(char in) {
		auto search = transition.find(in);
		if (search == transition.end()) {
			return make_pair(nullptr, -1);
		}
		else {
			return search->second;
		}
	}

	/// <summary>
	/// </summary>
	/// <returns> Assigned data for this state
	bool getData() {
		return data;
	}
};

class Scanner {
private:
	ifstream* fin;

	map<string, SingleState*> states;

	// state cursor
	SingleState* currentState;

	// token buffer
	string tokenBuffer;

	// flush token buffer on next process
	bool flushFlag;

	// recent error
	int errorType;

	typedef enum StateData {
		nonf = 0,
		fID, fNUM,
		fADD, fSUB, fMUL, fDIV,
		fLT, fLTE, fGT, fGTE, fEQ, fNEQ,
		fASSIGN, fENDS, fCOMMA,
		fLP, fRP, fLSB, fRSB, fLCB, fRCB
	};

	typedef enum TransitionOption {
		optNORMAL, optLOOKAHEAD, optDISCARD
	};

	void addState(string name, StateData data=nonf) {
		states[name] = new SingleState(data);
	}

	void mapState(string from, string to,
					string instring,
					TransitionOption opt=optNORMAL) {
		SingleState* s1 = states[from];
		SingleState* s2 = states[to];
		for (char in : instring) {
			s1->addMap(in, s2, opt);
		}
	}

	void buildState() {
		// Construct Automata
		addState("init");
		addState("ID");
		addState("NUM");
		addState("ADD", fADD);
		addState("SUB", fSUB);
		addState("MUL", fMUL);
		addState("DIV_COMMENT");
		addState("LT_LTE");
		addState("GT_GTE");
		addState("EQ_ASSIGN");
		addState("NEQ");
		addState("ENDS", fENDS);
		addState("COMMA", fCOMMA);
		addState("LP", fLP);
		addState("RP", fRP);
		addState("LSB", fLSB);
		addState("RSB", fRSB);
		addState("LCB", fLCB);
		addState("RCB", fRCB);
	}
	void buildInit() {
		mapState("init", "init", WHITESPACE, optDISCARD);
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
		addState("fID", fID);
		mapState("ID", "fID",
					NUMDIGIT+OTHERCHAR+WHITESPACE,
					optLOOKAHEAD);
	}
	void buildNUM() {
		mapState("NUM", "NUM", NUMDIGIT);
		addState("fNUM", fNUM);
		mapState("NUM", "fNUM",
					ALPHABET+OTHERCHAR+WHITESPACE,
					optLOOKAHEAD);
	}
	// processing: /, /**/
	void buildDIV_COMMENT() {
		addState("fDIV", fDIV);
		mapState("DIV_COMMENT", "fDIV",
					ALPHABET+NUMDIGIT+
						dropChars(OTHERCHAR, "*")
						+WHITESPACE,
					optLOOKAHEAD);

		addState("COMMENT");
		mapState("DIV_COMMENT", "COMMENT", "*");
		mapState("COMMENT", "COMMENT",
					ALPHABET+NUMDIGIT+
						dropChars(OTHERCHAR, "*")
						+WHITESPACE);	// ignore comments

		addState("COMMENTend");
		mapState("COMMENT", "COMMENTend", "*");
		mapState("COMMENTend", "init", "/");
	}
	// processing: <, <=
	void buildLT_LTE() {
		addState("fLT", fLT);
		mapState("LT_LTE", "fLT",
					ALPHABET+NUMDIGIT+
						dropChars(OTHERCHAR, "=")
						+WHITESPACE,
					optLOOKAHEAD);
		
		addState("fLTE", fLTE);
		mapState("LT_LTE", "fLTE", "=");
	}
	// processing: >, >=
	void buildGT_GTE() {
		addState("fGT", fGT);
		mapState("GT_GTE", "fGT",
					ALPHABET+NUMDIGIT+
						dropChars(OTHERCHAR, "=")
						+WHITESPACE,
					optLOOKAHEAD);
		
		addState("fGTE", fGTE);
		mapState("GT_GTE", "fGTE", "=");
	}
	// processing: ==, =
	void buildEQ_ASSIGN() {
		addState("fEQ", fEQ);
		mapState("EQ_ASSIGN", "fEQ", "=");

		addState("fASSIGN", fASSIGN);
		mapState("EQ_ASSIGN", "fASSIGN",
					ALPHABET+NUMDIGIT+
						dropChars(OTHERCHAR, "=")
						+WHITESPACE,
					optLOOKAHEAD);
	}
	void buildNEQ() {
		addState("fNEQ", fNEQ);
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
		terr = -1,
		tnull,
		tELSE, tIF, tINT, tRETURN, tVOID, tWHILE,
		tID, tNUM,
		tADD, tSUB, tMUL, tDIV, tLT, tLTE, tGT, tGTE, tEQ, tNEQ,
		tASSIGN, tENDS, tCOMMA,
		tLP, tRP, tLSB, tRSB, tLCB, tRCB
	};

	typedef enum ErrorType {
		eNOERROR,
		eNOMATCHINGTTYPE,
		eINVALIDINPUT,
		eENDOFFILE
	};

	Scanner(ifstream* f) : fin(f),
							tokenBuffer(""),
							flushFlag(false),
							errorType(eNOERROR) {
		// build DFA
		buildState();
		buildInit();
		buildNUM();
		buildDIV_COMMENT();
		buildLT_LTE();
		buildGT_GTE();
		buildEQ_ASSIGN();
		buildNEQ();

		currentState = states["init"];
	}

	TokenType processChar() {
		char in;

		// EOF?
		if (!(*fin >> in)) {
			errorType = eENDOFFILE;
			return terr;
		}

		// flush?
		if (flushFlag) {
			tokenBuffer.clear();
			flushFlag = false;
		}

		// retrieve result from current state
		auto temp = currentState->pushChar(in);
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

			return terr;
		}

		// set no error
		errorType = eNOERROR;

		// process transition option
		switch (opt) {
		case optLOOKAHEAD:	
			fin->putback(in);
			break;
		case optDISCARD:
			// discard
			break;
		case optNORMAL:
			tokenBuffer.push_back(in);
			break;
		default:
			tokenBuffer.push_back(in);
		}

		// process state data
		int stateData = nextState->getData();
		if (stateData == nonf) {
			return tnull;
		}
		else {
			flushFlag = true;
			currentState = states["init"];
			switch (nextState->getData()) {
			case fID:		return static_cast<TokenType>(findKeyword(tokenBuffer));
			case fNUM:		return tNUM;
			case fADD:		return tADD;
			case fSUB:		return tSUB;
			case fMUL:		return tMUL;
			case fDIV:		return tDIV;
			case fLT:		return tLT;
			case fLTE:		return tLTE;
			case fGT:		return tGT;
			case fGTE:		return tGTE;
			case fEQ:		return tEQ;
			case fNEQ:		return tNEQ;
			case fASSIGN:	return tASSIGN;
			case fENDS:		return tENDS;
			case fCOMMA:	return tCOMMA;
			case fLP:		return tLP;
			case fRP:		return tRP;
			case fLSB:		return tLSB;
			case fRSB:		return tRSB;
			case fLCB:		return tLCB;
			case fRCB:		return tRCB;
			default:
				errorType = eNOMATCHINGTTYPE;
				return terr;
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

