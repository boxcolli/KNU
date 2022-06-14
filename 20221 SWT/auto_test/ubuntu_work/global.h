#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>

using namespace std;

class CodeScanner {
public:
    CodeScanner(ifstream& f);
    string getdec() { return fdec; }
    string getname() { return fname; }
private:
    ifstream& f;
    string fline;   // raw declaration line
    string fdec;    // trimmed declaration line (no semi)
    string fname;   // tokenized function name
};

enum class DType {
	C, SC, UC,		// char
	S, US,			// short
	I, UI,			// int
	L, UL, LL, ULL,	// long
	F, D, LD,		// float
	Error
};

class CaseScanner {
public:
    enum OPT {
        CASE_RAW, CASE_LITERAL
    };
    CaseScanner(ifstream& f, OPT opt);
    bool next();
    string getfdec() { return fdec; }
    string getfname() { return fname; }
    vector<string> getargs() { return args; }
    string getret() { return ret; }
private:
    ifstream& f;
    CaseScanner::OPT opt;
    string fdec;
    string fname;
    vector<string> args;
    vector<DType> argstype;
    string ret;
    DType rettype;
    bool ignorecomment();
    string addAffix(string arg, DType dt);
};

class DriverPrinter {
public:
    enum OPT { D_NOSTUB, D_STUB };
    DriverPrinter(ofstream& f, string fdec, string fname, OPT opt);
    void printcase(vector<string> input, string output);
    void end();
private:
    ofstream& f;
    string fname;
    int count = 0;
    bool end_flag = false;
};

#endif