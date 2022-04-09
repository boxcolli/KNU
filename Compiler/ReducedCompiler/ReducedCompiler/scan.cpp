#include <fstream>
#include <map>
#include <string>

#define ALPHABET "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
			abcdefghijklmnopqrstuvwxyz"
#define NUMDIGIT "0123456789"
#define OTHERCHAR " \t\n`~!@#$%^&*()-_=+[]{};:'\""

using namespace std;

class SingleState {
private:
	/// <summary>
	/// Represents this state is non-final or final of which result.
	/// </summary>
	int data;

	/// <summary>
	/// Map holding transition data
	/// </summary>
	map<char, SingleState> transition;

public:
	/// <summary>
	/// Class SingleState constructor
	/// </summary>
	/// <param name="fin">Given true if this state is final, if else, false</param>
	SingleState(int data) : data(data) {}

	/// <summary>
	/// Add a transition from this state to another.
	/// </summary>
	/// <param name="in">A single input character</param>
	/// <param name="next">Pointer to another state</param>
	void addMap(char in, SingleState next) {
		transition[in] = next;
	}

	/// <summary>
	/// Push a single input character to make transition
	/// </summary>
	/// <param name="in">A single input character</param>
	/// <returns>nullptr if not found or mapped. SingleState* if found.</returns>
	SingleState* pushChar(char in) {
		auto search = transition.find(in);
		if (search == transition.end()) {
			return nullptr;
		}
		else {
			return &(search->second);
		}
	}

	/// <summary>
	/// </summary>
	/// <returns>True if final state, if else, false</returns>
	bool isFinal() {
		return data != 0;
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
	map<string, SingleState> states;

	SingleState* current;

	void addState(string name, int data) {
		states[name] = SingleState(data);
	}

	void mapState(string from, string to, string ins) {
		SingleState s1 = states[from];
		SingleState s2 = states[to];
		for (char in : ins) {
			s1.addMap(in, s2);
		}
	}

	void buildInit() {
		mapState("init", "ID", ALPHABET);
		mapState("init", "NUM", NUMDIGIT);
		mapState("init", "ADD", "+");
		mapState("init", "SUB", "-");
		mapState("init", "MUL", "*");
		mapState("init", "DIV", "/");
		mapState("init", "LT", "<");
		mapState("init", "GT", ">");
		mapState("init", "EQ", "=");
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

		addState("yesID", fID);
		
	}
	void buildNUM() {}
	void buildADD() {}
	void buildSUB() {}
	void buildMUL() {}
	void buildDIV() {}
	void buildLT() {}
	void buildGT() {}
	void buildEQ() {}
	void buildNEQ() {}
	void buildENDS() {}
	void buildCOMMA() {}
	void buildLP() {}
	void buildRP() {}
	void buildLSB() {}
	void buildRSB() {}
	void buildLCB() {}
	void buildRCB() {}

public:
	enum StateData {
		nonf = 0,
		fID, fNUM,
		fADD, fSUB, fMUL, fDIV,
		fLT, fLTE, fGT, fGTE, fEQ, fNEQ,
		fASSIGN, fENDS, fCOMMA,
		fLP, fRP, fLSB, fRSB, fLCB, fRCB
	};

	Scanner(ifstream* f) : fin(f), current(nullptr) {
		// Construct Automata
		addState("init", nonf);
		addState("ID", nonf);
		addState("NUM", nonf);
		addState("ADD", fADD);
		addState("SUB", fSUB);
		addState("MUL", fMUL);
		addState("DIV", nonf);	// w/ COMMENT
		addState("LT", nonf);	// w/ LTE
		addState("GT", nonf);	// w/ GTE
		addState("EQ", nonf);	// w/ ASSIGN
		addState("NEQ", fNEQ);
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
		buildADD();
		buildSUB();
		buildMUL();
		buildDIV();
		buildLT();
		buildGT();
		buildEQ();
		buildNEQ();
		buildENDS();
		buildCOMMA();
		buildLP();
		buildRP();
		buildLSB();
		buildRSB();
		buildLCB();
		buildRCB();
	}
};

