# first line: 1
@memory.cache
def listPolynomialtransAndParams():
    polytransformations = [("polytran", PolynomialFeatures())]
    polytransformationparameters = [[{"polytran__degree": [1,2,3]}]]
    return polytransformations, polytransformationparameters
