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
	/// True if final state, if else, false.
	/// </summary>
	bool finalState;

	/// <summary>
	/// Map holding transition data
	/// </summary>
	map<char, SingleState> transition;

public:
	/// <summary>
	/// Class SingleState constructor
	/// </summary>
	/// <param name="fin">Given true if this state is final, if else, false</param>
	SingleState(bool f) : finalState(f) {}

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
		return finalState;
	}
};

class MultiState {
	
};

class Scanner {
private:
	ifstream* fin;

	map<string, SingleState> states;

	SingleState* current;

	void addState(string name, bool finalState) {
		states[name] = SingleState(finalState);
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

		addState("yesID", false);
		
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
	Scanner(ifstream* f) : fin(f), current(nullptr) {
		// Construct Automata
		addState("init", false);
		addState("ID", false);
		addState("NUM", false);
		addState("ADD", false);
		addState("SUB", false);
		addState("MUL", false);
		addState("DIV", false);// w/ COMMENT
		addState("LT", false);	// w/ LTE
		addState("GT", false);	// w/ GTE
		addState("EQ", false); // w/ ASSIGN
		addState("NEQ", false);
		addState("ENDS", false);
		addState("COMMA", false);
		addState("LP", false);
		addState("RP", false);
		addState("LSB", false);
		addState("RSB", false);
		addState("LCB", false);
		addState("RCB", false);
		addState("yesToken", true);
		addState("noToken", true);

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