#include "globals.h"
#include "state.h"
#include "parse.h"
#include "fhead.h"
/**************************************************
Grammar Handler
**************************************************/
GrammarHandler::GrammarHandler(ifstream* f) {
    fHeader = FileHeader(f);

    buildStates();
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