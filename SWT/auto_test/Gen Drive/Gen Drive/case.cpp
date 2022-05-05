#ifndef _CASE_
#define _CASE_

#include "global.h"

CaseScanner::CaseScanner(ifstream& f) : f(f) {
	// ignore comments at the beginning
	while (true) {
		streampos pos = f.tellg();
		string line;
		char c;
		
		c = f.get();
		if (c == '/') {
			c = f.get();

			// enter comment '//'
			if (c == '/') {
				getline(f, line);
			}
			// enter comment '/*'
			else if (c == '*') {
				pos = f.tellg();
				// ...
			}
		}
	}
	
}

#endif