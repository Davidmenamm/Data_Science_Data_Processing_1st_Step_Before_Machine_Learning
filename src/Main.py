# Program that performs the data pre-processing step
# Pre-processing is needed before any ML model, can be runned
# Data sources are .test and .train (joined), spectf+heart
# From UCI datasets: http://archive.ics.uci.edu/ml/datasets/SPECTF+Heart
# This are complete datasets, no empty values, low dimension,
# low numerosity and binary output.
# Therefore, only normalization and reduction are applied.


# Imports
from Coordinator import Coordinator as Cd

# paths of the datasets
# input
trainPath = r'src\data\input\SPECTF.train'
testPath = r'src\data\input\SPECTF.test'
# output
joinedPath = r'src\data\output\1_joinedDataSet.txt'
normPath = r'src\data\output\2_normDataSet.txt'
rankPath = r'src\data\output\3_ranking.txt'
corrTabPath = r'src\data\output\4_correlationTable.txt'
reduceRankPath = r'src\data\output\5_reducedRanking.txt'
finalPath = r'src\data\output\6_reducedRanking.csv'


# function to write trazability of program to txt file
def writeToTxt(data, path):
    data = str(data)
    with open(path, 'w') as f:  # open the file to write
        f.write(data)


# coordinator instance to carry out all data processing, process
coord = Cd(6)  # num of decimals as parameter

# join both datasets, inputs are dataset paths
joined = coord.join(trainPath, testPath)
writeToTxt(joined, joinedPath)

# normalize joined dataSet
norm = coord.normalize(joined)
writeToTxt(norm, normPath)

# apply filter and extract ranking to data set (chi-squared)
rank = coord.runFilt(norm, 20)
writeToTxt(rank, rankPath)

# apply classifier (pearson correlation)
finalDataSet, corrTable = coord.runClassifier(rank, norm)
# correlation table
writeToTxt(corrTable, corrTabPath)

# print final result to output file (.csv) and txt
writeToTxt(finalDataSet, reduceRankPath)
finalDataSet.to_csv(
    finalPath, index=False)
