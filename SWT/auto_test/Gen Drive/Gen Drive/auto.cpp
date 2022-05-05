#ifndef _AUTO_
#define _AUTO_

#include "global.h"


AutoGenerator::AutoGenerator(ofstream& f, string funName, string funDec)
	: f(f), fName(funName), fDec(funDec), testEnd(false) {

	f << "#include <stdio.h>\n";
	f << funDec << endl;
	f << endl;
	f << "int main() {\n";
}

void AutoGenerator::genTest(int num, vector<string> args, string result) {
	f << "\tif (" << fName << "(";
	f << args[0];
	for (auto it = args.begin()+1; it != args.end(); it++) {
		f << ", " << *it;
	}
	f << ") == " << result << ") ";
	f << "printf(\"test case " << num << ": pass\\n\");" << endl;
	f << "else printf(\"test case " << num << ": Fail\\n\");" << endl;
	f << endl;
}

void AutoGenerator::endTest() {
	if (!testEnd) {
		testEnd = true;
		f << "\treturn 0;\n}";
	}
}

AutoGenerator::~AutoGenerator() {
	endTest();
}

#endif