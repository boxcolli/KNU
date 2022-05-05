#include "global.h"

template <typename T>
class TestCase {
private:

public:
};

int main(int argc, char** argv) {
    if (argc != 4) {
        cout << "usage: gendriver <code> <case> <driver>" << endl;
        return 0;
    }
    ifstream fcode(argv[1]);
    ifstream fcase(argv[2]);
    ofstream fdriver(argv[3]);

    // check argument, return type
    string fundec;
    getline(fcode, fundec);
    FunScanner funScanner(fundec);
    // TODO: if not valid function, exit


    // prepare driver file
    AutoGenerator autoGenerator(
        fdriver,
        funScanner.getName(),
        fundec
    );

    // read case file -> write driver file


}