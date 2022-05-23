#include "globals.h"
#include "scan.h"

const string ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const string NUMDIGIT = "0123456789";
const string OTHERCHAR = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const string WHITESPACE = " \t\n\v\f\r";

/// Returns a copy of the base string after erasing some characters.
string dropChars(const string base, string chars) {
	string s = base;
	for (auto it = s.begin(); it != s.end(); it++) {
		for (char c : chars) {
			if (*it == c) {
				s.erase(it--);
				break;
			}
		}
	}
	return s;
}

enum class TransitionOpt {
	optNORMAL, optLOOKAHEAD, optDISCARD
};

/******************************
Scanner::Public
******************************/
Scanner::Scanner(ifstream* f) :
    fileHeader(FileHeader(f)),
    tokenBuffer(""),
    flushFlag(false),
    errorType(TokErrType::eNOERROR) {
    
    // build DFA
    EOFSTR.push_back(EOF);
    buildState();
    buildInit();
    buildID();
    buildNUM();
    buildDIV_COMMENT();
    buildLT_LTE();
    buildGT_GTE();
    buildEQ_ASSIGN();
    buildNEQ();

    // init
    currentState = states["init"];
}

TokenType Scanner::processChar() {
    char in = fileHeader.getChar();
    
    newline = (in == '\n') ? true : false;  // newline?
    
    // flush token?
    if (flushFlag) {
        tokenBuffer.clear();
        flushFlag = false;
    }

    // pushChar to current state
    pair<SingleState*, TransitionOpt> temp = currentState->pushInput(in);
    SingleState* nextState = temp.first;
    TransitionOpt opt = temp.second;

    // no result?
    if (nextState == nullptr) {
        // character
        if (currentState == states["init"]) {
            if (in == EOF) {
                errorType = TokErrType::eENDOFFILE;
            }
            else {                
                errorType = TokErrType::eINVALIDINPUT;  // set error        
                tokenBuffer.push_back(in);              // retrieve char                
                flushFlag = true;                       // flush next time
            }
        }
        // comment not closed
        else if (currentState == states["COMMENT"]
            || currentState == states["COMMENTS2"]) {
            errorType = TokErrType::eNOCOMMENTEND;
        }
        // invalid REX rule
        else {            
            currentState = states["init"];          // force to reset DFA            
            errorType = TokErrType::eINVALIDRULE;   // set error            
            fileHeader.putBack(in);                 // re-process current input            
            flushFlag = true;                       // flush next time
        }
        return TokenType::tERROR;
    }

    // set no error
    errorType = TokErrType::eNOERROR;

    // process transition option
    switch (static_cast<TransitionOpt>(opt)) {
    case TransitionOpt::optLOOKAHEAD:
        fileHeader.putBack(in); break;
    case TransitionOpt::optDISCARD: // discard
        break;
    case TransitionOpt::optNORMAL:
        tokenBuffer.push_back(in); break;
    default:    
        tokenBuffer.push_back(in);
    }

    // get state data
    StateData stateData = static_cast<StateData>(nextState->getData());
    if (stateData == StateData::nonf) {
        // token not made
        currentState = nextState;
        return TokenType::tNULL;
    }
    else if (stateData == StateData::fCOMMENT) {
        // comment closed
        currentState = states["init"];
        flushFlag = true;
        return TokenType::tNULL;
    }
    else {
        // token has been made
        flushFlag = true;
        currentState = states["init"];
        return stateToToken(stateData);
    }
}

TokenType Scanner::processToken() {
    TokenType t;
    while ((t = processChar()) != TokenType::tNULL) {}
    return t;
}

/******************************
Scanner::Private
******************************/
void Scanner::mapState(string from, string to,
	string instring,
	TransitionOpt opt = TransitionOpt::optNORMAL) {
	// state 사이에 transition을 만듭니다.
	// 입력 문자를 코드 편의를 위해 스트링으로 한번에 처리합니다.
	SingleState* s1 = states[from];
	SingleState* s2 = states[to];
	for (char in : instring) {
		s1->addMap(in, s2, opt);
	}
}

void Scanner::buildState() {
	// Construct Automata
	addState("init");
	addState("ID");
	addState("NUM");
	addState("ADD", StateData::fADD);
	addState("SUB", StateData::fSUB);
	addState("MUL", StateData::fMUL);
	addState("DIV_COMMENT");
	addState("LT_LTE");
	addState("GT_GTE");
	addState("EQ_ASSIGN");
	addState("NEQ");
	addState("ENDS", StateData::fENDS);
	addState("COMMA", StateData::fCOMMA);
	addState("LP", StateData::fLP);
	addState("RP", StateData::fRP);
	addState("LSB", StateData::fLSB);
	addState("RSB", StateData::fRSB);
	addState("LCB", StateData::fLCB);
	addState("RCB", StateData::fRCB);
}
void Scanner::buildInit() {
    // init과 각 REX의 state 사이 transition을 잇습니다.
    mapState("init", "init", WHITESPACE, TransitionOpt::optDISCARD);
    mapState("init", "ID", ALPHABET);
    mapState("init", "NUM", NUMDIGIT);
    mapState("init", "ADD", "+");
    mapState("init", "SUB", "-");
    mapState("init", "MUL", "*");
    mapState("init", "DIV_COMMENT", "/");
    mapState("init", "LT_LTE", "<");
    mapState("init", "GT_GTE", ">");
    mapState("init", "EQ_ASSIGN", "=");
    mapState("init", "NEQ", "!");
    mapState("init", "ENDS", ";");
    mapState("init", "COMMA", ",");
    mapState("init", "LP", "(");
    mapState("init", "RP", ")");
    mapState("init", "LSB", "[");
    mapState("init", "RSB", "]");
    mapState("init", "LCB", "{");
    mapState("init", "RCB", "}");
}
void Scanner::buildID() {
    mapState("ID", "ID", ALPHABET);

    addState("fID", StateData::fID);
    mapState("ID", "fID",
        NUMDIGIT + OTHERCHAR,
        TransitionOpt::optLOOKAHEAD);
    mapState("ID", "fID",
        WHITESPACE + EOFSTR,
        TransitionOpt::optDISCARD);
}
void Scanner::buildNUM() {
    mapState("NUM", "NUM", NUMDIGIT);

    addState("fNUM", StateData::fNUM);
    mapState("NUM", "fNUM",
        ALPHABET + OTHERCHAR,
        TransitionOpt::optLOOKAHEAD);
    mapState("NUM", "fNUM",
        WHITESPACE + EOFSTR,
        TransitionOpt::optDISCARD);
}
// processing: /, /**/
void Scanner::buildDIV_COMMENT() {
    addState("fDIV", StateData::fDIV);
    mapState("DIV_COMMENT", "fDIV",
        ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "*"),
        TransitionOpt::optLOOKAHEAD);
    mapState("DIV_COMMENT", "fDIV",
        WHITESPACE + EOFSTR,
        TransitionOpt::optDISCARD);

    addState("COMMENT");
    mapState("DIV_COMMENT", "COMMENT", "*");
    mapState("COMMENT", "COMMENT",
        ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "*")
        + WHITESPACE,
        TransitionOpt::optDISCARD);	// ignore comments

    addState("COMMENT2");
    mapState("COMMENT", "COMMENT2", "*");
    mapState("COMMENT2", "COMMENT",
        ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "/")
        + WHITESPACE,
        TransitionOpt::optDISCARD);

    addState("COMMENTend", StateData::fCOMMENT);
    mapState("COMMENT2", "COMMENTend", "/",
        TransitionOpt::optDISCARD);
}
// processing: <, <=
void Scanner::buildLT_LTE() {
    addState("fLT", StateData::fLT);
    mapState("LT_LTE", "fLT",
        ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "="),
        TransitionOpt::optLOOKAHEAD);
    mapState("LT_LTE", "fLT",
        WHITESPACE + EOFSTR,
        TransitionOpt::optDISCARD);

    addState("fLTE", StateData::fLTE);
    mapState("LT_LTE", "fLTE", "=");
}
// processing: >, >=
void Scanner::buildGT_GTE() {
    addState("fGT", StateData::fGT);
    mapState("GT_GTE", "fGT",
        ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "="),
        TransitionOpt::optLOOKAHEAD);
    mapState("GT_GTE", "fGT",
        WHITESPACE + EOFSTR,
        TransitionOpt::optDISCARD);

    addState("fGTE", StateData::fGTE);
    mapState("GT_GTE", "fGTE", "=");
}
// processing: ==, =
void Scanner::buildEQ_ASSIGN() {
    addState("fEQ", StateData::fEQ);
    mapState("EQ_ASSIGN", "fEQ", "=");

    addState("fASSIGN", StateData::fASSIGN);
    mapState("EQ_ASSIGN", "fASSIGN",
        ALPHABET + NUMDIGIT + dropChars(OTHERCHAR, "="),
        TransitionOpt::optLOOKAHEAD);
    mapState("EQ_ASSIGN", "fASSIGN",
        WHITESPACE + EOFSTR,
        TransitionOpt::optDISCARD);
}
void Scanner::buildNEQ() {
    addState("fNEQ", StateData::fNEQ);
    mapState("NEQ", "fNEQ", "=");
}

TokenType Scanner::findKeyword(string token) {
    if (token == "else") return TokenType::tELSE;
    else if (token == "if") return TokenType::tIF;
    else if (token == "int") return TokenType::tINT;
    else if (token == "return") return TokenType::tRETURN;
    else if (token == "void") return TokenType::tVOID;
    else if (token == "while") return TokenType::tWHILE;
    else return TokenType::tID;
}

TokenType Scanner::stateToToken(StateData stateData) {
    switch (stateData) {
    case StateData::fID:		return findKeyword(tokenBuffer);
    case StateData::fNUM:		return TokenType::tNUM;
    case StateData::fADD:		return TokenType::tADD;
    case StateData::fSUB:		return TokenType::tSUB;
    case StateData::fMUL:		return TokenType::tMUL;
    case StateData::fDIV:		return TokenType::tDIV;
    case StateData::fLT:		return TokenType::tLT;
    case StateData::fLTE:		return TokenType::tLTE;
    case StateData::fGT:		return TokenType::tGT;
    case StateData::fGTE:		return TokenType::tGTE;
    case StateData::fEQ:		return TokenType::tEQ;
    case StateData::fNEQ:		return TokenType::tNEQ;
    case StateData::fASSIGN:	return TokenType::tASSIGN;
    case StateData::fENDS:		return TokenType::tENDS;
    case StateData::fCOMMA:		return TokenType::tCOMMA;
    case StateData::fLP:		return TokenType::tLP;
    case StateData::fRP:		return TokenType::tRP;
    case StateData::fLSB:		return TokenType::tLSB;
    case StateData::fRSB:		return TokenType::tRSB;
    case StateData::fLCB:		return TokenType::tLCB;
    case StateData::fRCB:		return TokenType::tRCB;
    default:
        errorType = TokErrType::eNOMATCHINGTTYPE;
        return TokenType::tERROR;
    }
}