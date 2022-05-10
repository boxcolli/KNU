#include "global.h"

void ERR_EXIT(string msg, int e) {
    cout << msg << endl;
    exit(e);
}

int main(int argc, char** argv) {
    if (argc < 3)
        ERR_EXIT("usage: gendriver [options] <case file> <output name>", 1);



    // pass options, get file names
    int argcount = 1;
    bool rawflag = true;
    for (; argcount != argc; argcount++) {
        if (argv[argcount][0] == '-') {
            // TODO: handle option
            string opt = argv[argcount];
            if (opt == "-raw" || opt == "-r") {
                rawflag = true;
            }
            else if (opt == "-literal" || opt == "-l") {
                rawflag = false;
            }
        }
        else {
            break;
        }
    }



    // open files
    if (argcount == argc)
        ERR_EXIT("usage: gendriver [options] <SUT file> <case file> <output name>", 1);
    /*
    ifstream fcode(argv[argcount++]);
    if (!fcode.is_open()) { ERR_EXIT("SUT file not open", 1); }
    */
    ifstream fcase(argv[argcount++]);
    if (!fcase.is_open()) { ERR_EXIT("case file not open", 1); }
    ofstream fdrive(argv[argcount]);
    if (!fdrive.is_open()) { ERR_EXIT("drive file not open", 1); }



    // prepare files
    //CodeScanner codeScanner(fcode);
    CaseScanner::OPT case_opt;
    if (rawflag)    case_opt = CaseScanner::CASE_RAW;
    else            case_opt = CaseScanner::CASE_LITERAL;
    CaseScanner caseScanner(fcase, case_opt);

    DriverPrinter driverPrinter(
        fdrive,
        caseScanner.getfdec(),
        caseScanner.getfname(),
        DriverPrinter::D_NOSTUB
    );



    // print cases
    while (caseScanner.next()) {
        driverPrinter.printcase(caseScanner.getargs(), caseScanner.getret());
    }



    // end driver
    driverPrinter.end();



    return 0;
}