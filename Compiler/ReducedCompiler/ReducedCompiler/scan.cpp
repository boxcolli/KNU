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
	/// <summary>
	/// File stream pointer
	/// </summary>
	ifstream* fin;

	/// <summary>
	/// 
	/// </summary>
	map<string, SingleState*> states;

	/// <summary>
	/// 
	/// </summary>
	SingleState* current;

	/// <summary>
	/// 
	/// </summary>
	enum StateData {
		nonf = 0,
		fID, fNUM,
		fADD, fSUB, fMUL, fDIV,
		fLT, fLTE, fGT, fGTE, fEQ, fNEQ,
		fASSIGN, fENDS, fCOMMA,
		fLP, fRP, fLSB, fRSB, fLCB, fRCB
	};
	enum TransitionOption {
		optNORMAL, optLOOKAHEAD
	};

	void addState(string name, int data=nonf) {
		states[name] = new SingleState(data);
	}

	void mapState(string from, string to, string instring, int opt=optNORMAL) {
		SingleState* s1 = states[from];
		SingleState* s2 = states[to];
		for (char in : instring) {
			s1->addMap(in, s2, opt);
		}
	}

	void buildInit() {
		mapState("init", "init", WHITESPACE);
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

public:
	
	/// <summary>
	/// Represents whether current input had made an enter to a final state,
	/// and if it is, represents which type of final state.
	/// </summary>
	enum TokenType {
		tnull,
		tELSE, tIF, tINT, tRETURN, tVOID, tWHILE,
		tADD, tSUB, tMUL, tDIV, tLT, tLTE, tGT, tGTE, tEQ, tNEQ,
		tASSIGN, tENDS, tCOMMA,
		tLP, tRP, tLSB, tRSB, tLCB, tRCB
	};

	Scanner(ifstream* f) : fin(f), current(nullptr) {
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

		buildInit();
		buildNUM();
		buildDIV_COMMENT();
		buildLT_LTE();
		buildGT_GTE();
		buildEQ_ASSIGN();
		buildNEQ();
	}
};

