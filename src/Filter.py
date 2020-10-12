# # In charge of building a rank to measure the attribute correlation
# # with the main attribute, which is the overall diagnosis

# # Imports
# import pandas as pd
# import numpy as np
# from sk.learn.feature_selection import SelectKBest, chi2


# # Class Filter
# class Filter:
#     # constructor
#     def __init__(self):
#         pass

#     # run filter, uses chi2 test
#     # returns rank for all attributes, relating to main attribute (binary)
#     def run(self, normDataSet):
#         # target col and other cols
#         targetCol = normDataSet.iloc[:, 0]
#         otherCol = normDataSet.iloc[:, 1:]
#         # get top 10 features, must pass the way of scoring function
#         topFeatures = SelectKBest(score_func=chi2, k=10)
#         topFeatures = topFeatures.fit()
