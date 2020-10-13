# To manage all other classes, with exception to Main.py
# and perform all data pre-processing

# Imports
import pandas as pd
from Filter import Filter as Ft
from Classifier import Classifier as Cf

# Coordinator Class


class Coordinator:
    # Constructor
    def __init__(self, decimals):
        if(decimals <= 12):
            self.decimals = decimals
        else:
            self.decimals = 12

    # to join datasets
    def join(self, pathA, pathB):
        dataSetA = pd.read_csv(pathA, delimiter=',')
        dataSetB = pd.read_csv(pathB, delimiter=',')
        joined = dataSetA.append(
            dataSetB, ignore_index=True)  # continuous idxs
        return joined

    # normalize columns (atributes), according to min-max technique
    def normalize(self, dataSet):
        # change pd dataframe data type to float
        cols = dataSet.columns
        for col in cols:
            dataSet[col] = dataSet[col].astype(float)

        # transpose dataSet (pd frame) and apply technique
        dataSet = dataSet.transpose()
        idxRow = 0
        for _, dim in dataSet.iterrows():  # for dimension or attributes
            min = dim.min()
            max = dim.max()
            for idxCol, value in dim.items():  # for idx or numerosity
                dataSet.iat[int(idxRow), int(idxCol)] = round(
                    (value-min)/(max-min), 6)
            idxRow += 1
        # to original position
        dataSet = dataSet.transpose()
        # return
        return dataSet

    # apply filter, get ranking for data set. Chi2 used.
    def runFilt(self, normDataSet, num):
        filt = Ft()
        filt = filt.runFilter(normDataSet, num)
        return filt

    # run classifier, uses pearson correlation
    # receives filtered dataSet
    # returns dataSet with low correlation atributes (columns)
    def runClassifier(self, filtDataSet, normDataSet):
        cf = Cf(self.decimals)
        lowCorrDataSet = cf.runClassifier(filtDataSet, normDataSet)
        return lowCorrDataSet
