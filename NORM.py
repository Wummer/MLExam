from __future__ import division
import numpy as np

"""
This function takes a dataset with class labels and computes the mean and the variance of each input feature 
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	mean = sum(data) / len(data)
	variance = sum((data - mean)**2) / len(data)
	return mean, variance
	


"""
This function calls mean_variance to get the mean and the variance from the dataset.
Then each datapoint is normalized using means and variance. 
"""
def meanfree(data):
	mean, variance = mean_variance(data)
	meanfree = (data - mean) / np.sqrt(variance)
	return meanfree



"""
This function transforms the test set using the mean and variance from the train set.
It expects the train and test set and call a function to get mean and variance of the train set.
It uses these to transform all datapoints of the test set. 
It returns the transformed test set. 

"""
def transformtest(trainset, testset):
	meantrain, variancetrain = mean_variance(trainset)
	transformed = (testset - meantrain) / np.sqrt(variancetrain)
	return transformed
