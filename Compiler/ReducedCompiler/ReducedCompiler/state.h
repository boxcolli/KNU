#ifndef _STATE_H_
#define _STATE_H_

class SingleState {
private:

	/// Can hold any optional data. For example, non-final, final or else.
	int data;

	/// Holds transition data, with transition option
	map<char, pair<SingleState*, int>> transition;

public:

	SingleState(int data = -1);

	/// Add a transition from this state to another.
	/// in: A single input character
	/// next: Pointer to another state
	void addMap(char in, SingleState* next, int opt);

	/// Push a single input character to make transition
	/// in: A single input character</param>
	/// returns nullptr if not found or mapped. SingleState* if found.
	pair<SingleState*, int> pushChar(char in);

	/// returns assigned data for this state.
	int getData();
};



#endif