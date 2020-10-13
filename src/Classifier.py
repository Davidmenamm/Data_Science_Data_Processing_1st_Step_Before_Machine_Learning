# In charge of comparing the correlation, feature to feature
# Leaves one atribute in a group of high correlation attributes

# Imports
import pandas as pd
from scipy.stats import pearsonr
from collections import defaultdict


# Class Filter
class Classifier:
    # Constructor
    def __init__(self, decimals):
        self.decimals = decimals

    # rescale proportions of list using min-max method
    def rescale(self, ls):
        # last case of the list
        if len(ls) == 1:
            ls[0] = 1.0
        # other cases of the list
        else:
            minValue = min(ls)
            maxValue = max(ls)
            for idx in range(0, len(ls)):
                ls[idx] = round((ls[idx]-minValue) /
                                (maxValue-minValue), self.decimals)
        # return
        return ls

    # Build a correlation table
    # Receives a vector of the correlation of one elm with the others
    def corrTable(self, compVectDF, colNames):
        dictCorr = defaultdict(list)
        colLength = len(colNames)
        for idxA in range(0, colLength):
            name = colNames[idxA]
            # first element correlation with others
            if idxA == 0:
                # add to dictionary
                dictCorr[name].extend(compVectDF)
            # other elements correlation with others
            else:
                relationVector = []
                # compare proportions with other elms
                for idxB in range(0, colLength):
                    val1 = compVectDF[idxA]
                    val2 = compVectDF[idxB]
                    minValue = min(val1, val2)
                    maxValue = max(val1, val2)
                    relationVector.append(
                        round(minValue/maxValue, self.decimals))
                # add to dictionary
                dictCorr[name].extend(relationVector)
        # Dictionary to dataframe
        dfCorrTable = pd.DataFrame.from_dict(dict(dictCorr), orient='index')
        # return
        return dfCorrTable.transpose()

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
        minPercent = 0.85
        # Array to store correlations
        lsColumns = []  # list of dataSet Columns (series)
        lsCorr = []  # list of correlations
        colLength = len(filtDataSet.columns)

        # First column in the dataSet
        firstQntCol = filtDataSet.iloc[:, 0]
        # other columns
        for idx in range(0, colLength):
            # current col
            currentCol = filtDataSet.iloc[:, idx]
            # calculate correlation respective to first quantitative column
            correlation = pearsonr(firstQntCol, currentCol)  # tuple value

            # test print
            if idx == 0:
                print('correlation', correlation)

            # append columns to ls, as a dataframe
            currentCol = pd.DataFrame({colNames[idx]: currentCol.values})
            lsColumns.append(currentCol)
            # append correlation to ls
            lsCorr.append(correlation[0])

        # get correlation table, only for visualization purposes
        corrTab = self.corrTable(lsCorr, colNames)

        # Conserve only columns with low correlation between them
        # From a high correlation group, store only one col
        while len(lsColumns) > 0 and len(lsCorr) > 0:
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
            # Rescale proportions of remaining elmnts in correlations list]
            if len(lsCorr) > 0:
                lsCorr = self.rescale(lsCorr)

        # transform arrOfSeries (columns), into one dataframe
        finalDataFrame = lowCorrColumns[0]
        for idx in range(1, len(lowCorrColumns)):
            finalDataFrame = pd.concat(
                [finalDataFrame, lowCorrColumns[idx]], axis=1)
        # return
        return finalDataFrame, corrTab
