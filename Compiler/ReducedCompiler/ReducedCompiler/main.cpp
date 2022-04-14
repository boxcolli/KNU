#include "globals.h"
#include "scan.h"

#include <list>

typedef struct ScanResult {
    TokenType ttype = TokenType::tNULL;
    TokErrType etype = TokErrType::eNOERROR;
    string name = "";
} ScanResult;

static string fname = "2.c";

int main() {
    ifstream f_scan(fname);
    Scanner scanner(&f_scan);
    TokenType tresult;
    
    ifstream f_print(fname);
    string buffer;
    if (!(getline(f_print, buffer))) {
        exit(1);
    }

    int linecount = 1;
    list<ScanResult> linetokens;

    bool loop = true;
    bool forceprint = false;
    while (loop) {
        // process one char
        tresult = scanner.processChar();




        // check token result
        if (tresult == TokenType::tERROR) {
            // error
            TokErrType e = scanner.getErrorType();
            if (e == TokErrType::eENDOFFILE) {
                // print final line and end
                loop = false;
                forceprint = true;
            }
            else if (e == TokErrType::eNOCOMMENTEND) {
                // print final line with error and end
                loop = false;
                forceprint = true;
                ScanResult r;
                r.ttype = tresult;
                r.etype = e;
                r.name = scanner.getToken();
                linetokens.push_back(r);
            }
            else {
                // print error and continue
                ScanResult r;
                r.ttype = tresult;
                r.etype = e;
                r.name = scanner.getToken();
                linetokens.push_back(r);
            }
        }
        else if (tresult != TokenType::tNULL) {
            // a token has been made
            ScanResult r;
            r.ttype = tresult;
            r.name = scanner.getToken();
            linetokens.push_back(r);
        }





        // check newline
        if (scanner.isNewLine() || forceprint) {
            forceprint = false;

            // print out line
            cout << linecount << ": " << buffer << endl;


            // print out token list
            for (auto r : linetokens) {
                cout << "\t" << linecount << ": ";
                switch (r.ttype) {
		        case TokenType::tID:
                    cout << "ID" << ", name= " << r.name << endl; break;
                case TokenType::tELSE:
                    cout << "reserved word: else" << endl; break;
                case TokenType::tIF:
                    cout << "reserved word: if" << endl; break;
                case TokenType::tINT:
                    cout << "reserved word: int" << endl; break;
                case TokenType::tRETURN:
                    cout << "reserved word: return" << endl; break;
                case TokenType::tVOID:
                    cout << "reserved word: void" << endl; break;
                case TokenType::tWHILE:
                    cout << "reserved word: while" << endl; break;
		        case TokenType::tNUM:
                    cout << "NUM" << ", val= " << stoi(r.name) << endl; break;
		        case TokenType::tADD:
                    cout << "+" << endl; break;
		        case TokenType::tSUB:
                    cout << "-" << endl; break;
		        case TokenType::tMUL:
                    cout << "*" << endl; break;
		        case TokenType::tDIV:
                    cout << "/" << endl; break;
		        case TokenType::tLT:
                    cout << "<" << endl; break;
		        case TokenType::tLTE:
                    cout << "<=" << endl; break;
		        case TokenType::tGT:
                    cout << ">" << endl; break;
		        case TokenType::tGTE:
                    cout << ">=" << endl; break;
		        case TokenType::tEQ:
                    cout << "==" << endl; break;
		        case TokenType::tNEQ:
                    cout << "!=" << endl; break;
		        case TokenType::tASSIGN:
                    cout << "=" << endl; break;
		        case TokenType::tENDS:
                    cout << ";" << endl; break;
		        case TokenType::tCOMMA:
                    cout << "," << endl; break;
		        case TokenType::tLP:
                    cout << "(" << endl; break;
		        case TokenType::tRP:
                    cout << ")" << endl; break;
		        case TokenType::tLSB:
                    cout << "[" << endl; break;
		        case TokenType::tRSB:
                    cout << "]" << endl; break;
		        case TokenType::tLCB:
                    cout << "{" << endl; break;
		        case TokenType::tRCB:
                    cout << "}" << endl; break;
		        default:
                    // error
			        switch (r.etype) {
                    case TokErrType::eNOMATCHINGTTYPE:
                        cout << "error: no matching token type found: " << r.name << endl; break;
                    case TokErrType::eINVALIDINPUT:
                        cout << "error: invalid input: " << r.name << endl; break;
                    case TokErrType::eNOCOMMENTEND:
                        cout << "error: missing comment end '*/'" << endl; break;
                    case TokErrType::eINVALIDRULE:
                        cout << "error: incomplete token: " << r.name << endl; break;
                    default:
                        cout << "error: unknown" << endl; break;
                    }
		        }
            }
            linetokens.clear();
            linecount++;
            getline(f_print, buffer);
        }
    }
    

    
    return 0;
}