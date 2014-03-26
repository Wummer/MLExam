from __future__ import division
import numpy as np
#-----------------------------------------------------------------------
#Computing mean and variance
"""
This function takes a dataset and computes the mean and the variance of each input feature (leaving the class column out)
It returns two lists:
[mean of first feature, mean of second feature]
[variance of first feature, variance of second feature]
"""
def mean_variance(data):
	Mean = []
	Variance = []
	number_of_features = len(data[0])
	for i in xrange(number_of_features): 
		s = 0
		su = 0

		#mean
		for elem in data:
			s +=elem[i]
		mean = s / len(data)
		Mean.append(mean)
		
		#variance:
		for elem in data:
			su += (elem[i] - Mean[i])**2
			variance = su/len(data)	
		Variance.append(variance)
	return Mean, Variance




"""
This function calls mean_variance to get the mean and the variance for each feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy thus preserving class label 
The new normalized set is then returned.
"""
def meanfree(data):
	for e in data:
		number_of_features = len(e) - 1 #Leaving out the class

	mean, variance = mean_variance(data)

	new = np.copy(data)

	for num in xrange(number_of_features):
		for i in xrange(len(data)):
			r = (data[i][num] - mean[num]) / np.sqrt(variance[num])
			new[i][num] = r #replacing at correct index in the copy
	return new


def transformtest(trainset, testset):
	#getting the mean and variance from train:
	meantrain, variancetrain = mean_variance(trainset)

	number_of_features = len(trainset[0]) - 1 #Leaving out the class

	newtest = np.copy(testset)

	for num in xrange(number_of_features):
		for i in xrange(len(testset)):
			#replacing at correct index in the copy
			newtest[i][num] = (testset[i][num] - meantrain[num]) / np.sqrt(variancetrain[num])
	return newtest

