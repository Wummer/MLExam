from __future__ import division
import numpy as np
import pylab as plt


"""
Here we sample the mean from a distribution. 
It returns a matrix with each number denoting the mean in the specific dimension
"""
def MLmean(data):
	c1 = np.zeros(10)
	for d in data:
		c1 += d

	return c1/len(data)


"""
Here we sample the covariance from the distribution.
It returns a single covariance matrix that reflects the distribution.
"""
def MLcov(data,ML):
	samples = []
	nM  = 0

	for d in data:
		n = d-ML
		nM += np.outer(n,n)
	CML = 1/len(data)*nM

	return CML

"""
Here we project the data onto the specified principal components.
This is done by calculating the dot product between the PC and the data points.
The new 2D coordinates are returned.

"""
def transform(data,components):
	new_x= []
	new_y= []
	for d in data:
		d = d.reshape(10,1)
		new_x.append(np.dot(components[0],d))
		new_y.append(np.dot(components[1],d))

	return new_x,new_y