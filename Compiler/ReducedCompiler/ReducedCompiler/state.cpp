#include "globals.h"
#include "state.h"

template<class T>
class ListState : FiniteState {
private:
	vector<T> l;
public:
	ListState(T l);
};