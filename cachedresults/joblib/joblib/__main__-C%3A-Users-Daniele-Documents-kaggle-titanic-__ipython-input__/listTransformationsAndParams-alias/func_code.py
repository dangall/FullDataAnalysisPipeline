# first line: 1
@memory.cache
def listTransformationsAndParams():
    transformations = [("sqrt", FunctionTransformer(sqrttransform)), 
                       ("log", FunctionTransformer(logtransform)), 
                       ("scale", StandardScaler())]
    transformationparameters = [[{}],[{}],[{}]]
    return transformations, transformationparameters
