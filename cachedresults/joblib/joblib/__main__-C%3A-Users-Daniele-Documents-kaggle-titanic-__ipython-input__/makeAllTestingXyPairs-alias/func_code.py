# first line: 1
@memory.cache
def makeAllTestingXyPairs(thedataframe, topredict, frompredict, dictofordinals, datacateg, whichcombinations="all"):
    predictionframe = thedataframe[thedataframe[topredict]!=dictofunknown[topredict]]
    allframestotry = makeAllTestingDataframes(predictionframe, dictofordinals, datacateg, whichcombinations=whichcombinations)
    Xypairs = [(turnFrameIntoMatrix(df[frompredict]), turnFrameIntoMatrix(df[[topredict]])) for df in allframestotry]
    return Xypairs
