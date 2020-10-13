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

    # rescale proportions of list using min-max method
    def rescale(self, ls):
        if len(ls) > 0:
            minValue = min(ls)
            maxValue = max(ls)
            for idx in range(0, len(ls)):
                ls[idx] = (ls[idx]-minValue)/(maxValue-minValue)
        # return
        return ls

    # Run classifier, uses pearson correlation
    # Returns low correlation dataSet, receives filtered dataSet
    def runClassifier(self, filtDataSet, normDataSet):
        # transpose data set for correct addressing, on correct dataSet
        filtDataSet.set_index('Attribute', inplace=True)
        filtDataSet = filtDataSet.transpose()
        colNames = list(filtDataSet)
        filtDataSet = normDataSet[colNames]
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
            # append columns to ls, as a dataframe

            currentCol = pd.DataFrame({colNames[idx]: currentCol.values})
            lsColumns.append(currentCol)
            # append correlation to ls
            lsCorr.append(correlation[0])

        # Conserve only columns with low correlation between them
        # From a high correlation group, store only one col
        while len(lsColumns) > 0 and len(lsCorr) > 1:
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
            # Rescale proportions of remaining elmnts in correlations list
            lsCorr = self.rescale(lsCorr)

        # transform arrOfSeries (columns), into one dataframe
        finalDataFrame = lowCorrColumns[0]
        for idx in range(1, len(lowCorrColumns)):
            finalDataFrame = pd.concat(
                [finalDataFrame, lowCorrColumns[idx]], axis=1)
        # return
        return finalDataFrame
