#ifndef _SCAN_
#define _SCAN_

#include "global.h"

typedef enum class KW {
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

FunScanner::FunScanner(string funDecLine) : fundec(funDecLine) {
	// init tokenizer
	char* str = new char[fundec.length() + 1];
	strcpy(str, fundec.c_str());
	char delim[] = " (,)";
	char* tok = strtok(str, delim);



	// 1. return type
	vector<KW> ret_klist;
	while (tok) {		
		KW k = mapKeyw(tok);		// string -> enum
		tok = strtok(str, delim);	// prepare next token
		if (k == KW::NIL) break;	// not keyword (function name)
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
	vector<KW> arg_klist;
	while (tok) {
		KW k = mapKeyw(tok);

		if (k == KW::NIL) {
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