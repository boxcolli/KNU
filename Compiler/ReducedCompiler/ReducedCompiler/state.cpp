#include "globals.h"
#include "state.h"

SingleState::SingleState(int data = -1) : data(data) {}

void SingleState::addMap(char in, SingleState* next, int opt) {
	transition[in] = make_pair(next, opt);
}

pair<SingleState*, int> SingleState::pushChar(char in) {
	auto search = transition.find(in);
	if (search == transition.end()) {
		return make_pair(nullptr, -1);
	}
	else {
		return search->second;
	}
}

int SingleState::getData() {
	return data;
}