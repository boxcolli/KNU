#ifndef _STATE_H_
#define _STATE_H_

#include "globals.h"

template <typename Data, typename In, typename Opt>
class FiniteState {
private:

	/// Can hold any optional data. For example, non-final, final or else.
	Data data;

	/// Holds transition data, with transition option
	map<char, pair<FiniteState*, Opt>> transition;

public:

	FiniteState(Data data) : data(data) {}

	/// Add a transition from this state to another.
	/// in: A single input character
	/// next: Pointer to another state
	void addMap(In in, FiniteState* next, Opt opt) {
		transition[in] = make_pair(next, opt);
	}

	/// Push a single input character to make transition
	/// in: A single input character</param>
	/// returns nullptr if not found or mapped. FiniteState* if found.
	pair<FiniteState*, int> pushInput(In in) {
		auto search = transition.find(in);
		if (search == transition.end()) {
			return make_pair(nullptr, -1);
		}
		else {
			return search->second;
		}
	}

	/// returns assigned data for this state.
	DT getData() { return data; }
};

#endif