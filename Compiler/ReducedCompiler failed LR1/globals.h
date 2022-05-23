#ifndef _GLOBALS_H_
#define _GLOBALS_H_

#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <set>
using namespace std;

// Parser
#define EPSILON "EMPTY"         // nullable terminal
#define START_SYM "<program>"   // starting symbol
#define END_SYM "$"             // end/last/bottom symbol

void MSG_EXIT(string msg);

#endif