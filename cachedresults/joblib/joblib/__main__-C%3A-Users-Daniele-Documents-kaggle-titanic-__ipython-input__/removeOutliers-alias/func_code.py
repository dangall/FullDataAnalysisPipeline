# first line: 1
@memory.cache
def removeOutliers(inputX, inputy):
    # We first find outliers using the method IsolationForest
    forest = IsolationForest()
    forest.fit(inputX)
    predictedforest = forest.predict(inputX)
    # We now find outliers using OneClassSVM
    svm = OneClassSVM()
    svm.fit(inputX)
    predictedsvm = svm.predict(inputX)
    # When the two methods agree on an outlier, it is an outlier and we remove it
    predicted = predictedforest + predictedsvm
    posnonoutlier = np.where(predicted>-2)[0]
    if list(posnonoutlier)==range(len(X)):
        outputX, outputy = True,True
    else:
        outputX, outputy = inputX[posnonoutlier], inputy[posnonoutlier]
    return outputX, outputy
