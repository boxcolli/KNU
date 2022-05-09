#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>

#include <sstream>
#include <algorithm>
#include <iterator>

using namespace std;

class CodeScanner {
private:
    ifstream& f;
    string fline;   // raw declaration line
    string fdec;    // trimmed declaration line (no semi)
    string fname;   // tokenized function name
public:
    CodeScanner(ifstream& f);
    string getdec() { return fdec; }
    string getname() { return fname; }
};

class CaseScanner {
private:
    ifstream& f;

public: 
    CaseScanner(ifstream& f) : f(f) {}
    bool process();
};

class DriverPrinter {
private:

public:

};

#endif