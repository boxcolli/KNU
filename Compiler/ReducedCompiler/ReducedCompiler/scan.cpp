#include <fstream>
#include <map>
#include <string>

#define ALPHABET "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
			abcdefghijklmnopqrstuvwxyz"
#define NUMDIGIT "0123456789"

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

class Scanner {
private:
	ifstream* fin;

	map<string, SingleState> states;

	SingleState* current;

public:
	Scanner(ifstream* f) : fin(f), current(nullptr) {
		// Construct Automata
		addState("init", SingleState(false));
		addState("ID", SingleState(false));
		addState("NUM", SingleState(false));
		addState("ADD", SingleState(false));
		addState("SUB", SingleState(false));
		addState("MUL", SingleState(false));
		addState("DIV", SingleState(false));
		addState("LT", SingleState(false));	// w/ LTE
		addState("GT", SingleState(false));	// w/ GTE
		addState("EQ", SingleState(false)); // w/ ASSIGN
		addState("NEQ", SingleState(false));
		addState("ENDS", SingleState(false));
		addState("COMMA", SingleState(false));
		addState("LP", SingleState(false));
		addState("RP", SingleState(false));
		addState("LSB", SingleState(false));
		addState("RSB", SingleState(false));
		addState("LCB", SingleState(false));
		addState("RCB", SingleState(false));
		addState("COMMENT", SingleState(false));

		/// init ///
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
		mapState("init", "COMMENT", "/");

		/// ID ///
		
	}

	void addState(string name, SingleState state) {
		states[name] = state;
	}

	void mapState(string from, string to, string ins) {
		SingleState s1 = states[from];
		SingleState s2 = states[to];
		for (char in : ins) {
			s1.addMap(in, s2);
		}		
	}

	
};