# In charge of comparing the correlation, feature to feature
# Leaves one atribute in a group of high correlation attributes

# Imports
import pandas as pd
from scipy.stats import pearsonr


# Class Filter
class Classifier:
    # Constructor
    def __init__(self):
        pass

    # Run classifier, uses pearson correlation
    # Returns low correlation dataSet, receives filtered dataSet
    def runClassifier(self, filtDataSet):
        # To store final low correlation columns
        lowCorrColumns = []
        # Min percent accepted for it to be a correlation
        minPercent = 0.9
        # Array to store correlations
        lsColumns = []  # list of dataSet Columns (series)
        lsCorr = []  # list of correlations
        colLength = len(filtDataSet.columns)

        # First quantitative column in the dataSet (idx)
        firstQntCol = filtDataSet.iloc[:, 1]

        # Quantitive cols are considered; in other words not 1st binary col
        for idx in range(1, colLength):
            # current col
            currentCol = filtDataSet.iloc[:, idx]
            # calculate correlation respective to first quantitative column
            correlation = pearsonr(firstQntCol, currentCol)  # tuple value
            # append columns to ls
            lsColumns.append(currentCol)
            # append correlation to ls
            lsCorr.append(correlation[0])

        # Conserve only columns with low correlation between them
        # From a high correlation group, store only one col
        while len(lsColumns) > 0:
            # store correlated idxs, each loop
            corrIdxs = []
            for idx in range(0, len(lsCorr)):
                if(lsCorr[idx] > minPercent):
                    corrIdxs.append(idx)
            # store only one of the correlated columns
            toStore = lsColumns[corrIdxs[0]]
            lowCorrColumns.append(toStore)
            # idx revised, are removed from col and corr lists
            lsColumns = [i for j, i in enumerate(
                lsColumns) if j not in corrIdxs]
            lsCorr = [i for j, i in enumerate(
                lsCorr) if j not in corrIdxs]
        # return
        # return ranking
