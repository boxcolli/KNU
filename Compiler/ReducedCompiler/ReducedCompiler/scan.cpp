#include <fstream>
#include <map>
#include <string>

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
		addState(string("init"), SingleState(false));
		addState(string("init"), SingleState(false));
	}

	void addState(string name, SingleState state) {
		states[name] = state;
	}

	void mapState(string from, string to, char in) {
		SingleState s1 = states[from];
		SingleState s2 = states[to];
		s1.addMap(in, s2);
	}

};