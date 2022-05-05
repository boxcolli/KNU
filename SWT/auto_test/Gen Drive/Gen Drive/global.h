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
	string getName() { return funName; }
	vector<DType> getArgType() { return argType; }
};

class AutoGenerator {
private:

	ofstream& f;
	string fName;
	string fDec;
	bool testEnd;

public:

	AutoGenerator(ofstream& f, string funName, string funDec);
	void genTest(int num, vector<string> args, string result);
	void endTest();
	~AutoGenerator();
};

class CaseScanner {
private:

	ifstream& f;
	vector<string> args;
	string result;

public:

	CaseScanner(ifstream& f);
	bool get();
	vector<string> getArgs() { return args; }
	string getResult() { return result; }


};

#endif