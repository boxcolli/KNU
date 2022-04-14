


TokenType processChar () {
    in ← getChar ()
    detectNewLine (&newline, in)
    flushBufferIf (flushFlag)

    nextState, opt ← pushCharToDFA (in)
    if nextState==nullptr then
    errorType, flushFlag ← examineError (currentState, in)
    return ERROR

    processTransitionOption (opt, in)

    stateData ← nextState.getData ()
    tokenType, currentState, flushFlag ← examineNextState (stateData)
    return tokenType
}