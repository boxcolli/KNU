#include "globals.h"
#include "state.h"
#include "parse.h"
#include "fhead.h"
/**************************************************
Grammar Handler
**************************************************/
const string SYMBOLCHAR = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-";
const string NUMDIGIT = "0123456789";
const string WHITESPACE = " \t\v\f\r";
const string NEWLINE = "\n";

enum class _BNFTransOpt {
    
};

GrammarHandler::GrammarHandler(ifstream* f) : fHeader(FileHeader(f)){
    buildStates();

    addState("init");

    addState("LS_begin", StateData::lbegin);
    addState("LS_mid");
    addState("LS_end", StateData::lend);

    addState("assign1");
    addState("assign2");
    addState("assign3");

    addState("RS_begin", StateData::rbegin);
    addState("RS_mid");
    addState("RS_end", StateData::rend);
    
    // <LSymbol>
    mapState("init", "init", WHITESPACE);
    mapState("init", "L_begin", "<");
    mapState("LS_begin", "LS_mid", SYMBOLCHAR);
    mapState("LS_mid", "LS_mid", SYMBOLCHAR);
    mapState("LS_mid", "LS_end", ">");
    mapState("LS_end", "LS_end", WHITESPACE);
    // :=
    mapState("LS_end", "assign1", ":");
    mapState("assign1", "assign2", ":");
    mapState("assign2", "assign3", "=");
    // <RSymbol>
    mapState("assign3", "assign3", WHITESPACE);    
    mapState("assign2", "RS_begin", "<");
    mapState("RS_begin", "RS_mid", SYMBOLCHAR);
    mapState("RS_mid", "RS_mid", SYMBOLCHAR);
    mapState("RS_mid", "RS_end", ">");
    mapState("RS_end", "RS_end", WHITESPACE);
    // RTerm
    mapState("")
    // |
    mapState("R_end", "R_")
}
void GrammarHandler::buildStates() {
    
}
/**************************************************
First Follow
**************************************************/
FirstFollow::FirstFollow() {

}
/**************************************************
Parser
**************************************************/
class Parser {
private:



public:



};