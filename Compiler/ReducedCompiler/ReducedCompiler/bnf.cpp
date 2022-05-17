#include "globals.h"
#include "bnf.h"
#include "state.h"
#include "fhead.h"

/**************************************************
BNF Scanner
**************************************************/
const string ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const string NUMDIGIT = "0123456789";
const string WHITESPACE = " \t\v\f\r";
const string NEWLINE = "\n";

enum class _BNFTransOpt {
    
};

BNFScanner::BNFScanner(ifstream* f) : fHeader(FileHeader(f)), tokenBuffer("") {
    addState("init");
    addState("symbol_begin");
    addState("symbol_mid");
    addState("symbol_end", TType::symbol);
    addState("term_begin");
    addState("term_end", TType::term);
    addState("assign_0");
    addState("assign_1");
    addState("assign_2", TType::assign);
    addState("or", TType::or);

    mapState("init", "init", WHITESPACE);
    // symbol
    mapState("init", "symbol_begin", "<");
    mapState("symbol_begin", "symbol_mid", ALPHABET + "-");
    mapState("symbol_mid", "symbol_mid", ALPHABET + "-" + NUMDIGIT);
    mapState("symbol_mid", "symbol_end", ">");
    // terminal
    mapState("init", "term_begin", ALPHABET);
    mapState("term_begin", "term_begin", ALPHABET + "-" + NUMDIGIT);
    mapState("term_begin", "term_end", WHITESPACE);
    // assign
    mapState("init", "assign_0", ":");
    mapState("assign_0", "assign_1", ":");
    mapState("assign_1", "assign_2", "=");
    // or
    mapState("init", "or", "|");
    // member
    initState = states["init"];
    currentState = initState;
}

BNFScanner::TType BNFScanner::pushInput() {
    char in = fHeader.getChar();

    // newline?
    if (in == '\n') { return TType::newline; }
    // EOF?
    if (in == EOF) { return TType::eof; }

    // get next state
    auto p = currentState->pushInput(in);
    auto state = p.first;    
    if (state == nullptr) {
        MSG_EXIT("BNF : invalid rule at..." + fHeader.getBuffer());
    }

    // token += input
    if (state == initState) { return TType::empty; }
    tokenBuffer += in;

    // ignore some state
    auto data = static_cast<TType>(state->getData());
    if (data == TType::empty) {  return TType::empty; }

    // return token
    switch (data) {    
    case TType::symbol:
    case TType::term:
    case TType::assign:
    case TType::or:
        return data;
    default:
        MSG_EXIT("BNF : invalid rule at..." + fHeader.getBuffer());
    }
}

/**************************************************
BNF Parser
**************************************************/