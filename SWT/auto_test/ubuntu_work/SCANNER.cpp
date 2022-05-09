#ifndef _SCANNER_
#define _SCANNER_

#include "GLOBAL.h"

string keywords[] = {
    "signed", "unsigned",
    "char", "short", "int", "long", "flaot", "double",
    "void",
    "static" 
};

CodeScanner::CodeScanner(ifstream& f) : f(f) {
    getline(f, fline);

    /*
        get fdec : trimmed declaration
    */
   int index = 1;
   for (char c : fline) {
       if (c == ')') break; // find ')' location
       index++;
   }
   fdec = fline.substr(0, index);

    /*
        get fname : function name
    */
    char* str = new char[fline.length() + 1];
    strcpy(str, fline.c_str());
    char delim[] = " (";
    char* tok = strtok(str, delim);

    // ignore return type keywords    
    while (true) {
        bool found = true;

        if (tok == NULL) {
            cout << "SUT function declaration not found" << endl;
            exit(1);
        }

        for (string kw : keywords) {
            if (kw == tok) {
                found = false; // the token is keyword
                break;
            }
        }

        if (found) break;

        tok = strtok(NULL, delim);
    }

    fname = tok;
}

bool CaseScanner::process() {
    while (!f.eof()) {
        // try to ignore comments
        char c; auto pos = f.tellg(); f >> c;
        if (c == '/') {
            f >> c;            
            if (c == '*') {
                // in comment state
                f >> c;
                while (true) {
                    if (c == '*') {
                        f >> c;
                        if (c == '/') {
                            break;
                        }
                    }
                    f >> c;
                }
                string temp;
                getline(f, temp);

                continue;
            }
            else {
                // restore and handle case;
                f.seekg(pos);                
                break;
            }
        }        
    }

    // tokenize test case
    string line;
    getline(f, line);
    string delim = " ";
    vector<string> tokens;
    auto start = 0U;
    auto end = line.find(delim);
    while (end != string::npos) {
        tokens.push_back(line.substr(start, end - start));
        start = end + delim.length();
        end = line.find(delim, start);
    }
    tokens.push_back(line.substr(start, end));
}

#endif