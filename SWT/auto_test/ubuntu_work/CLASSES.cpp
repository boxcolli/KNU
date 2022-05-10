#include "GLOBAL.h"

#define WHITESPACE " \t\n\v\f\r"

string keywords[] = {
    "signed", "unsigned",
    "char", "short", "int", "long", "flaot", "double",
    "void",
    "static" 
};

vector<string> tokenize(string s, string delim) {
    vector<string> tokens;

    char* str = new char[s.length() + 1];
    strcpy(str, s.c_str());
    char* del = new char[delim.length() + 1];
    strcpy(del, delim.c_str());

    char* tok = strtok(str, del);

    while (tok != NULL) {
        tokens.push_back(tok);     
        tok = strtok(NULL, del);
    }

    return tokens;
}

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

        if (tok == NULL)
            ERR_EXIT("SUT function declaration not found", 1);

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

bool CaseScanner::next() {
    while (true) {
        // try to ignore comments
        char c; auto pos = f.tellg();
        f >> c; if (f.eof()) return false;


        if (c == '/') {
            f >> c; if (f.eof()) return false;            


            if (c == '*') {
                // in comment state
                f >> c; if (f.eof()) return false;


                while (true) {
                    if (c == '*') {
                        f >> c; if (f.eof()) return false;


                        if (c == '/') {
                            break;
                        }
                    }


                    f >> c; if (f.eof()) return false;
                }


                string temp;
                getline(f, temp);
                continue;
            }
        }   

        // restore and handle case;
        f.seekg(pos);
        break;     
    }

    // tokenize test case
    string line;
    getline(f, line);
    cout << "line : " << line << endl;
    if (line == "") return false;

    vector<string> tokens = tokenize(line, WHITESPACE);

    // store tokens
    ret = tokens.back();
    //if (ret.back() == '\n') ret.pop_back();

    tokens.pop_back();
    args = tokens;

    // success
    return true;
}

DriverPrinter::DriverPrinter(ofstream& f, string fdec, string fname, OPT opt)
    : f(f), fname(fname) {
    f << "#include <stdlib.h>" << endl;
    f << "#include <stdio.h>" << endl;
    f << "#include <string.h>" << endl;
    f << endl;
    f << fdec << ";" << endl;
    f << endl;
    f << "int main() {" << endl;
    
}

void DriverPrinter::printcase(vector<string> input, string output) {
    // if (fname(...) == output)
    auto it = input.begin();
    f << "\t" << "if (" << fname << "(" << *it++;
    while (it != input.end()) f << ", " << *it++;
    f << ") == " << output << ") ";

    // printf("test case {count}: pass\n");
    f << "printf(\"test case " << count << " : pass\\n\");" << endl;

    // else printf("test case {count} : Fail\n");
    f << "\t" << "else printf(\"test case " << count++ << " : Fail\\n\");" << endl;
    f << endl;
}

void DriverPrinter::end() {
    if (!end_flag) {
        end_flag = true;
        f << "\t" << "return 0;" << endl;
        f << "}";
    }
}