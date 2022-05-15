#ifndef _STATE_H_
#define _STATE_H_

#include "globals.h"

template <typename Data, typename In, typename Opt, typename State>
class FiniteState {
public:

	FiniteState(Data data) : data(data) {}

	/// Add a transition from this state to another.
	/// in: A single input character
	/// next: Pointer to another state
	void addMap(In in, State* next, Opt opt) {
		transition[in] = make_pair(next, opt);
	}

	/// Push a single input character to make transition
	/// in: A single input character</param>
	/// returns nullptr if not found or mapped. FiniteState* if found
	pair<State*, Opt> pushInput(char in) {
		auto search = FiniteState::transition.find(in);
		if (search == transition.end()) {
			// not found
			return make_pair(nullptr, static_cast<Opt>(-1));
		}
		else {
			// return value
			return search->second;
		}
	}

	/// returns assigned data for this state.
	Data getData() { return data; }
	
private:

	/// Can hold any optional data. For example, non-final, final or else.
	Data data;

	/// Holds transition data, with transition option
	map<In, pair<State*, Opt>> transition;

};

#endif