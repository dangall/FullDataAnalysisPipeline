# first line: 1
@memory.cache
def listReductionsAndParams(estimator, totnumfeatures, predictiontype="classification"):
    if predictiontype=="regression":
        dimreductions = [("kbest", SelectKBest(mutual_info_regression)),
                         ("fromforest", SelectFromModel(RandomForestRegressor(n_estimators=150)))]
    else:
        dimreductions = [("kbest", SelectKBest(mutual_info_classif)), 
                         ("fromforest", SelectFromModel(RandomForestClassifier(n_estimators=150)))]
    
    dimreductions = dimreductions + [("rfecv", RFECV(estimator, cv=crossvalidator)), 
                                     ("frommodel", SelectFromModel(estimator)), 
                                     ("isomap", Isomap()), #Maintains geodesic distances between nearby points, 
                                                           #"unravelling" the whole manifold
                                     ("lle", LocallyLinearEmbedding(eigen_solver ="dense")), #Quite similar to Isomap, though uses a
                                                                        #different algorithm
                                     # FOR CLUSTERING
                                     #("spectral", SpectralEmbedding()), #Takes the dominant eigenvectors of the Laplacian matrix
                                     #("mds", MDS()), #Is good at preserving the distances (or their hierarchy) 
                                     #                #of the high-dim space
                                     #("tsne", TSNE()), #Is good at preserving local structure - it focuses on the 
                                     #                  #details of the high-dim space. It can deal with multiple, 
                                     #                  #disjoint manifolds
                                     ("pca", PCA()), 
                                     ("kernelpca", KernelPCA()), 
                                     ("truncsvd", TruncatedSVD()), 
                                     #("dictlearn", DictionaryLearning()),  #TOO SLOW
                                     ("factoranalys", FactorAnalysis())
                                    ]
    
    numfeaturestotry = range(1,totnumfeatures)
    neighbornumberstotry = range(1,11)
    dimredparameters = [
        [{'kbest__k': numfeaturestotry}], #SelectKBest
        [{}], #SelectFromModel(forest)
        [{}], #RFECV
        [{}], #SelectFromModel
        [{"isomap__n_neighbors": neighbornumberstotry, "isomap__n_components": numfeaturestotry}], #Isomap
        [{"lle__n_neighbors": neighbornumberstotry, "lle__n_components": numfeaturestotry}], #LocallyLinearEmbedding
        #[{"spectral__n_neighbors": neighbornumberstotry, "spectral__affinity": ["nearest_neighbors"], 
        #  "spectral__n_components": numfeaturestotry}, 
        # {"spectral__affinity": ["rbf"], "spectral__n_components": numfeaturestotry}], #SpectralEmbedding
        #{"mds__n_components": numfeaturestotry, "mds__metric": [True, False]}, #MDS
        #{"tsne__n_components": numfeaturestotry}, #TSNE
        [{"pca__n_components": numfeaturestotry}], #PCA
        [{"kernelpca__n_components": numfeaturestotry, "kernelpca__kernel": 
         ["linear", "poly", "rbf", "sigmoid", "cosine"]}], #KernelPCA
        [{"truncsvd__n_components": numfeaturestotry[:-1]}], #TruncatedSVD
        #{"dictlearn__n_components": numfeaturestotry}, #DictionaryLearning
        [{"factoranalys__n_components": numfeaturestotry}] #FactorAnalysis
    ]
    
    if predictiontype=="classification":
        dimreductions = dimreductions + [("lda", LinearDiscriminantAnalysis())]
        dimredparameters = dimredparameters + [[{"lda__n_components":numfeaturestotry, 
                                                 "lda__solver ": ["svd", "lsqr", "eigen"]}]]
    return dimreductions, dimredparameters
