# FullDataAnalysisPipeline
The notebooks provided here constitute a full Data Analysis pipeline, which starts from raw collected data and ends with predictions, regressions, classifications, and clustering. To exemplify how to use it, I ran predictions on the famous Titanic survival dataset https://www.kaggle.com/c/titanic and achieved a test-set score of about 83%.

The full process is divided into 3 main steps.

### Data Cleaning

This is done in `CleaningNotebook.ipynb`. This is a guided data cleaning notebook, which includes an analysis of the different data types and is aimed at catching bad entries, outliers, inconsistent types of data, etc. This notebook will then output a clean dataframe.

### Data Exploration

The notebook `DataExploration.ipynb` will provide extensive data visualization and statistical description of the data types. It begins with a global data analysis on the full dataset, and continues by fully analyzing all relevant segmentations of the data. Finally, it looks for all correlations between the various features but stops short of predicting.

### Predictions

Finally, the most complex notebook is `Predictions.ipynb`. Here I guide the user through the process of making an accurate prediction from the data, by transforming and engineering the various features of the data and trying a wide variety of predictive and clustering models on the data. The notebook is largely divided in three sections: 
 - a data preprocessing part, which generates many different versions of the data;
 - a prediction part, which attempts a given prediction on the various versions of the data to find the most suitable one;
 - a section evaluating the various predictions, by including a wide variety of metrics such as scores, ROC curves, confusion matrices, probability evaluations, and model running times.
 
 This notebook also outputs a new csv file with the predictions.

## Additional files

 - I also included a useful notebook, `GetNationality.ipynb`, which takes a dataframe with people's names and associated a nationality to each of those names, by scraping name-genealogy sites to identify the most common country for that name to appear in.
 - The csv files in included here are the various forms the data took along its path from raw to cleaned up to including new feature, and finally to including predictions.
 
 ![alt tag](http://personalpages.to.infn.it/~dgalloni/subimage/mecastle.jpg)
 
 There
