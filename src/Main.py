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
trainPath = r'src\data\SPECTF.train'
testPath = r'src\data\SPECTF.test'

# coordinator instance to carry out all data processing, process
coord = Cd()

# join both datasets, inputs are dataset paths
joined = coord.join(trainPath, testPath)

# normalize joined dataSet
norm = coord.normalize(joined)

# apply filter and extract ranking to data set
ranking = coord.applyFilt(norm, 10)

print('Ranking Top 10:\n', ranking)
