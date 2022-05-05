#ifndef _SCAN_
#define _SCAN_

#include "global.h"

typedef enum class Keyw {
	NIL,
	SIG, USIG,
	C, S, I, L, F, D,
	V,
	Count // to determine number of elements
};

Keyw mapKeyw(string tok) {
	if (tok == "signed") return Keyw::SIG;
	if (tok == "unsigned") return Keyw::USIG;
	if (tok == "char") return Keyw::C;
	if (tok == "short") return Keyw::S;
	if (tok == "int") return Keyw::I;
	if (tok == "long") return Keyw::L;
	if (tok == "float") return Keyw::F;
	if (tok == "double") return Keyw::D;
	if (tok == "void") return Keyw::V;
	return Keyw::NIL;
}

int ktoi(Keyw k) {
	return static_cast<int>(k);
}

DType mapDType(vector<Keyw> klist) {
	// count keyword
	vector<int> kcount(ktoi(Keyw::Count), 0);
	for (Keyw k : klist) {
		kcount[ktoi(k)]++;
	}



	// map
	if (kcount[ktoi(Keyw::C)] == 1) {
		if (kcount[ktoi(Keyw::SIG)] == 1) return DType::SC;
		if (kcount[ktoi(Keyw::USIG)] == 1) return DType::UC;
		return DType::C;
	}

	if (kcount[ktoi(Keyw::S)] == 1) {
		if (kcount[ktoi(Keyw::USIG)] == 1) return DType::US;
		return DType::S;
	}

	if (kcount[ktoi(Keyw::I)] == 1) {
		if (kcount[ktoi(Keyw::USIG)] == 1) {
			if (kcount[ktoi(Keyw::L)] == 1) return DType::UL;
			if (kcount[ktoi(Keyw::L)] == 2) return DType::ULL;
			return DType::UI;
		}
		if (kcount[ktoi(Keyw::L)] == 1) return DType::L;
		if (kcount[ktoi(Keyw::L)] == 2) return DType::LL;
		return DType::I;
	}

	if (kcount[ktoi(Keyw::F)] == 1) return DType::F;

	if (kcount[ktoi(Keyw::D)] == 1) {
		if (kcount[ktoi(Keyw::L)] == 1) return DType::LD;
		return DType::D;
	}

	if (kcount[ktoi(Keyw::L)] == 1) return DType::L;

	if (kcount[ktoi(Keyw::L)] == 2) return DType::LL;

	return DType::Error;
}

FunScanner::FunScanner(string funDecLine) : fundec(funDecLine) {
	// init tokenizer
	char* str = new char[fundec.length() + 1];
	strcpy(str, fundec.c_str());
	char delim[] = " (,)";
	char* tok = strtok(str, delim);



	// 1. return type
	vector<Keyw> ret_klist;
	while (tok) {		
		Keyw k = mapKeyw(tok);		// string -> enum
		tok = strtok(str, delim);	// prepare next token
		if (k == Keyw::NIL) break;	// not keyword (function name)
		ret_klist.push_back(k);		// flag[enum]++
	}
	retType = mapDType(ret_klist);



	// 2. function name
	if (tok != nullptr) {
		funName = string(tok);
	}
	else {
		// error
	}	
	tok = strtok(str, delim);



	// 3. arg type
	vector<Keyw> arg_klist;
	while (tok) {
		Keyw k = mapKeyw(tok);

		if (k == Keyw::NIL) {
			argType.push_back(mapDType(arg_klist));
			arg_klist.clear();
		}
		else {
			arg_klist.push_back(k);
		}

		tok = strtok(str, delim);
	}
}

#endif