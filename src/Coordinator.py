# To manage all other classes, with exception to Main.py
# and perform all data pre-processing

# Imports
import pandas as pd

# Coordinator Class


class Coordinator:
    # Constructor
    def __init__(self):
        pass

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
                dataSet.iat[int(idxRow), int(idxCol)] = (value-min)/(max-min)
            idxRow += 1
        # to original position
        dataSet = dataSet.transpose()
        # return
        return dataSet
