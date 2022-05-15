/**************************************************
File read helper	
**************************************************/

#ifndef _FHEAD_H_
#define _FHEAD_H_

#include "globals.h"

class FileHeader {
private:
	ifstream* fin;
	string buffer;
	size_t length;
	int cursor;

	char putBackChar = '\0';
	bool putBackFlag = false;

public:
	FileHeader(ifstream* fin) : fin(fin) {
		getline(*fin, buffer);
		length = buffer.length();
		cursor = 0;
	}

	char getChar() {
		// have pushed back char
		if (putBackFlag) {
			putBackFlag = false;
			return putBackChar;
		}

		// buffer empty?
		if (cursor == length) {
			buffer = "";
			getline(*fin, buffer);

			// EOF?
			if (buffer == "" && fin->eof()) {
				return EOF;
			}

			length = buffer.length();
			cursor = 0;
			return '\n';
		}

		return buffer[cursor++];
	}

	void putBack(char c) {
		putBackChar = c;
		putBackFlag = true;
	}
};

#endif