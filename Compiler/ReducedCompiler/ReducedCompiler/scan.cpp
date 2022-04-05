#include <map>

using namespace std;

enum class SingleResult {
	error, success
};

class SingleState {
private:
	bool fstate;
	map<char, SingleState*> table; // input -> nextState mapping
public:
	SingleState(bool f) : fstate(f) {}

	void addMap(char in, SingleState* next) {
		table.insert(make_pair(in, next));
	}

	int input(char in) {
		auto search = table.find(in);
		if (search == table.end()) {

		}
	}

	bool isFinal() {
		return fstate;
	}
};

class Automata {

};