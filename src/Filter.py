# In charge of building a rank to measure the attribute correlation
# with the main attribute, which is the overall diagnosis

# Imports
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2


# Class Filter
class Filter:
    # constructor
    def __init__(self):
        pass

    # run filter, uses chi2 test
    # returns rank for all attributes, relating to main attribute (binary)
    def runFilter(self, normDataSet, num):
        # target col and other cols
        targetCol = normDataSet.iloc[:, 0]
        otherCol = normDataSet.iloc[:, 1:]
        # get top 10 features, must pass the way of scoring function
        topFeatures = SelectKBest(score_func=chi2, k=10)
        topFeatures = topFeatures.fit(otherCol, targetCol)
        # extract scores & column names, to build a new pd frame ranking
        colNames = pd.DataFrame(otherCol.columns)
        scores = pd.DataFrame(topFeatures.scores_)
        ranking = pd.concat([colNames, scores], axis=1)
        # put column names, for ranking pd frame
        ranking.columns = ['Attribute', 'Score']
        # select top 10
        ranking = ranking.nlargest(num, 'Score')
        # return
        return ranking
