#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring> // strtok

using namespace std;

typedef enum class DType {
	C, SC, UC,		// char
	S, US,			// short
	I, UI,			// int
	L, UL, LL, ULL,	// long
	F, D, LD,		// float
	Error
};

class FunScanner {
private:

	string fundec;
	DType retType;
	string funName;
	vector<DType> argType;

public:
	
	FunScanner(string funDecLine);
	DType getRetType() { return retType; }
	vector<DType> getArgType() { return argType; }
};

#endif