#include "global.h"

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

enum class KW {
	NIL,
	SIG, USIG,
	C, S, I, L, F, D,
	V,
	Count // to determine number of elements
};

KW mapKeyw(string tok) {
	if (tok == "signed") return KW::SIG;
	if (tok == "unsigned") return KW::USIG;
	if (tok == "char") return KW::C;
	if (tok == "short") return KW::S;
	if (tok == "int") return KW::I;
	if (tok == "long") return KW::L;
	if (tok == "float") return KW::F;
	if (tok == "double") return KW::D;
	if (tok == "void") return KW::V;
	return KW::NIL;
}

int ktoi(KW k) {
	return static_cast<int>(k);
}

DType mapDType(vector<KW> klist) {
	// count keyword
	vector<int> kcount(ktoi(KW::Count), 0);
	for (KW k : klist) {
		kcount[ktoi(k)]++;
	}



	// map
	if (kcount[ktoi(KW::C)] == 1) {
		if (kcount[ktoi(KW::SIG)] == 1) return DType::SC;
		if (kcount[ktoi(KW::USIG)] == 1) return DType::UC;
		return DType::C;
	}

	if (kcount[ktoi(KW::S)] == 1) {
		if (kcount[ktoi(KW::USIG)] == 1) return DType::US;
		return DType::S;
	}

	if (kcount[ktoi(KW::I)] == 1) {
		if (kcount[ktoi(KW::USIG)] == 1) {
			if (kcount[ktoi(KW::L)] == 1) return DType::UL;
			if (kcount[ktoi(KW::L)] == 2) return DType::ULL;
			return DType::UI;
		}
		if (kcount[ktoi(KW::L)] == 1) return DType::L;
		if (kcount[ktoi(KW::L)] == 2) return DType::LL;
		return DType::I;
	}

	if (kcount[ktoi(KW::F)] == 1) return DType::F;

	if (kcount[ktoi(KW::D)] == 1) {
		if (kcount[ktoi(KW::L)] == 1) return DType::LD;
		return DType::D;
	}

	if (kcount[ktoi(KW::L)] == 1) return DType::L;

	if (kcount[ktoi(KW::L)] == 2) return DType::LL;

	return DType::Error;
}

DType getDType(vector<string> tokens) {
    vector<KW> keywords;
    for (string tok : tokens) {
        keywords.push_back(mapKeyw(tok));
    }
    return mapDType(keywords);
}

bool CaseScanner::ignorecomment() {
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

        // restore and break
        f.seekg(pos);
        break;
    }

    return true;
}

string CaseScanner::addAffix(string arg, DType dt) {
    string result = "";
    switch (dt) {
    case DType::C:
    case DType::SC:
    case DType::UC:
        result = "'" + arg + "'"; break;
    case DType::S:
    case DType::I:        
    case DType::F:
    case DType::D:
    case DType::LD:
        result = arg; break;
    case DType::US:
    case DType::UI:
        result = arg + "u"; break;
    case DType::L:
        result = arg + "l"; break;
    case DType::UL:
        result = arg + "ul"; break;
    case DType::LL:
        result = arg + "ll"; break;
    case DType::ULL:
        result = arg + "ull"; break;
    default:
        result = arg;
    }
    
    return result;
}

CaseScanner::CaseScanner(ifstream& f, OPT opt) : f(f) {
    // ignore comment
    if (!ignorecomment()) {
        // EOF
        cout << "invalid test case file" << endl;
        exit(1);
    }

    // process function declaration
    getline(f, fdec);
    vector<string> tokens = tokenize(fdec, " (,)");
    
    // process function return type
    // CASE_LITERAL : store keywords
    // CASE_RAW : ignore keywords
    auto it = tokens.begin();
    vector<string> kwtokens;
    while (true) {
        bool iskw = false;
        for (string kw : keywords) {
            if (*it == kw) { iskw = true; break; }
        }

        if (iskw) {
            if (opt == CASE_LITERAL)
                kwtokens.push_back(*it);
            it++;
            continue;
        }
        else
            break;
    }

    if (opt == CASE_LITERAL)
        rettype = getDType(kwtokens);        
    else
        rettype = DType::Error;
    
    

    // process function name
    fname = *it++;


    // process function arguments
    while (it != tokens.end()) {
        kwtokens.clear();
        while (true) {
            bool iskw = false;
            for (string kw : keywords)
                if (*it == kw) { iskw = true; break; }

            if (iskw) {
                if (opt == CASE_LITERAL)
                    kwtokens.push_back(*it);
                    it++;
                continue;
            }                
            else
                break;
        }

        if (opt == CASE_LITERAL) {
            argstype.push_back(getDType(kwtokens));            
        }
        else
            argstype.push_back(DType::Error);
        
        it++;
    }
}

bool CaseScanner::next() {
    if (!ignorecomment()) {
        // EOF
        return false;
    }

    // tokenize test case
    string line;
    getline(f, line);
    if (line == "") return false;

    vector<string> tokens = tokenize(line, WHITESPACE);
    tokens.erase(tokens.begin());    

    // store tokens
    if (opt == OPT::CASE_RAW) {
        ret = tokens.back();
        tokens.pop_back();
        args = tokens;       
    }
    else {
        ret = addAffix(tokens.back(), rettype);
        tokens.pop_back();

        args.clear();
        auto it_tok = tokens.begin();
        auto it_at = argstype.begin();
        for (int i = 0; i < tokens.size(); i++) {            
            args.push_back(addAffix(*it_tok, *it_at));
            it_tok++;
            it_at++;
        }
    }

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