# first line: 1
@memory.cache(ignore=['quiet'])
def bestEstimatorGridSearch(X, y, predictionmethod, paramstotest, predictiontype, quiet=False):
    modelparameters = deepcopy(paramstotest)
    if type(modelparameters)!=list:
        modelparameters = [modelparameters]
    # First we list the possible transformations
    transformations, transformationparameters = listTransformationsAndParams()
    # Now we list the possible dimensional reductions and their hyperparameters
    totnumfeatures = len(X[0])
    dimreductions, dimredparameters = listReductionsAndParams(predictionmethod[-1][1], totnumfeatures, 
                                                              predictiontype=predictiontype)
    if getModelName(predictionmethod[-1][1]) in dir(sklearn.linear_model):
        polytransformations, polytransformationparameters = listPolynomialtransAndParams()
    else:
        polytransformations, polytransformationparameters = [], []
    
    # We need to break up our dataset into a validation set and a set we use to find the best predictionmethod
    X_touse, X_finaltest , y_touse, y_finaltest = train_test_split(X, y, test_size=0.2)
    
    # We list all transformations, including the empty one (i.e. no transformation) in a way
    # that makes it easy to concatenate in a pipeline
    alltrans = [transformations[ii-1:ii] for ii in range(len(transformations)+1)]
    alldimreds = [dimreductions[ii-1:ii] for ii in range(len(dimreductions)+1)]
    allpolytrans = [polytransformations[ii-1:ii] for ii in range(len(polytransformations)+1)]
    # allgrids will contain the results of the fitted results from GridSearchCV of each pipeline
    allgrids = []
    for ti, trans in enumerate(alltrans):
        for di, dimred in enumerate(alldimreds):
            for pi, poly in enumerate(allpolytrans):
                #We need to add [[{}]] to transformationparameters because alltrans also contains the empty set
                paramstrans = ([[{}]] + transformationparameters)[ti] 
                paramsdimred = ([[{}]] + dimredparameters)[di]
                paramspoly = ([[{}]] + polytransformationparameters)[pi]
                paramstouse = [dict(dict1.items() + dict2.items() + dict3.items() + dict4.items()) 
                               for dict1 in paramstrans 
                               for dict2 in paramsdimred 
                               for dict3 in paramspoly 
                               for dict4 in modelparameters]
                estimatorchain = trans + dimred + poly + predictionmethod
                if quiet==False:
                    print [el[0] for el in estimatorchain]
                pipe = Pipeline(estimatorchain)
                gridobject = GridSearchCV(pipe, param_grid=paramstouse, cv=crossvalidator, error_score=np.nan)
                print gridobject
                gridobject.fit(X_touse,y_touse.ravel())
                allgrids.append(gridobject)
    # We are now done fitting all the individual pipelines
    maxscore = np.max([grid.best_score_ for grid in allgrids])
    bestmethod = [grid.best_estimator_ for grid in allgrids if grid.best_score_==maxscore][0]
    # We now have the best pipeline version of the model! Let's get a more accurate score on it
    bestscore = np.concatenate((cross_val_score(bestmethod, X_touse, y_touse.ravel(), cv=crossvalidator),
                                cross_val_score(bestmethod, X_touse, y_touse.ravel(), cv=crossvalidator)))
    bestscore = np.mean(bestscore), np.std(bestscore)
    # We will now test our GridSearch estimator on our validation test to check that it isn't much worse
    validationscore = bestmethod.score(X_finaltest, y_finaltest.ravel())
    isvalidationscoreworse = (validationscore < (bestscore[0] - 2.0*bestscore[1]))
    if isvalidationscoreworse==True and quiet==False:
        print "Final validation score is considerably worse than that claimed by GridSearchCV (problem of multiple comparisons)."
    # We can now finally get an accurate score of this method on the whole dataset
    finalscore = np.mean(np.concatenate((cross_val_score(bestmethod, X, y.ravel(), cv=crossvalidator),
                                     cross_val_score(bestmethod, X, y.ravel(), cv=crossvalidator))))
    return (allgrids, bestmethod, finalscore)
