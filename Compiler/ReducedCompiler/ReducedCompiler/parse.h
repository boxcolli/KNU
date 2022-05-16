#ifndef _PARSE_H_
#define _PARSE_H_

#include "globals.h"
#include "state.h"
#include "fhead.h"

typedef struct _rule {
	string lhand;
	vector<string> rhand;
};
typedef vector<_rule> _grammar;

/**************************************************
Grammar Handler
**************************************************/
class GrammarHandler {
public:
	GrammarHandler();

	

private:
	
};
/**************************************************
First Follow
**************************************************/
class FirstFollow {
public:
	FirstFollow();
private:
	// map term
	// map nont

	// map 

	// map first
	// map follow
	
};
/**************************************************
Parser
**************************************************/
class LR1Parser {
public:
	LR1Parser();
private:
	
};

#endif