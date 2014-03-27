from __future__ import division
import numpy as np
import pylab as plt
import sys

""" REGRESSION """

""" MAXIMUM LIKELIHOOD SOLUTION """

"""
Create the design matrix from vector X - either for linear or polynomial basis functions
- Linear design matrices would look like this: [1, x1, ..., xn]
- Polynomial design matrices would look like this: (e.g. with 3 variables)
[1, x1, x2, x3, x1^n, x2^n, x3^n, x1*x2, x1*x3, x2*x3] for each row 
"""
def createDesignMatrix(X, model="linear",degree=2):
	#if model=="linear, create an array of size 1+n, 
	#else if model=="quadratic", create an array of 1 + n + n + (summation of X0..Xn-1)
	l = len(X[0])
	size = 1 + l

	if model=="polynomial":
		size += l + (((l-1)*l) / 2)

	phi = np.zeros((len(X), size))

	#set first column values to 1
	phi[:,0] = 1 

	for c in range(l):
		phi[:,c+1] = X[:,c] # phi(x) = x

		if model=="polynomial":
			phi[:,c+1+l] = X[:,c] ** degree # phi(x) = x**n

	if model=="polynomial":
		# phi(x) = x1 * x2 ... for (n-1)!
		j = 0
		for c in xrange(0, l-1):
			for i in xrange(c+1, l):
				phi[:, j+1+l+l] = X[:, c] * X[:, i]
				j += 1

	return phi


""" 
Finding the Maximum Likelihood 
phi = design matrix PHI 
t = vector of corresponding target variables 
"""
def findML(phi, t):
	wML = np.dot(phi.T, phi)
	wML = np.linalg.pinv(wML)
	wML = np.dot(wML, phi.T)
	wML = np.dot(wML, t)
	return wML

""" 
In this function we will use the weight vectors from the training set and use them on the test set's variables
w = weight vectors from the train dataset 
X_test = the subset of X variables from the test set 
"""
def predict(w, X_test,model,degree):
	phi = createDesignMatrix(X_test,model,degree) #create design matrix from the test variables
	y = np.zeros(len(phi)) # predicted classes

	#summate over all rows in phi and save class vector in y
	for i in xrange(len(phi)):
		summation = 0
		for j in xrange(len(phi[i])):
			summation += w[j] * phi[i][j]
		y[i] = summation

	return y

""" 
This function calculates the Mean Square: 1/N * sum(tn-yn)
t = actual value from the dataset
y = predicted value 
"""
def calculateMS(t, y):
	N = len(t)
	summation = 0
	
	for n in xrange(N):
		summation += (t[n] - y[n])**2

	RMS = summation/N
	return RMS


""" MAXIMUM A POSTERIORI SOLUTION """
def computeBayesianMeanAndCovariance(X, t, alpha):
	beta = 1
	#get design matrix
	phi = createDesignMatrix(X)

	#get second part of covariance matrix
	bpp = beta * np.dot(phi.T, phi)

	#get first part of covariance matrix
	aI = np.zeros(bpp.shape)
	np.fill_diagonal(aI, alpha) #alpha * I

	covariance = aI + bpp
 	
 	#get each part of the mean equation
 	bs = beta * np.linalg.pinv(covariance)
 	mean = np.dot(bs, np.dot(phi.T, t))
 	mean = mean.reshape(-1,len(mean))[0]

 	return mean, covariance  


""" 
MAIN
"""
def run(train,test,method="ml",model="linear",degree=2,alphas=[0,1]):

	#Getting the vectors of all target variables
	t_train = train[:,:-1]
	t_test = test[:,:-1]

	#Extracting the class information
	c_train = train[:,-1]
	c_test = test[:,-1]

	if method=="ml":
		#Building design matrix
		phi_train = createDesignMatrix(t_train,model,degree)
		phi_test = createDesignMatrix(t_test,model,degree)

		#Getting weight vectors
		w = findML(phi_train, c_train)
		print w[0]

		#Getting the predicted target classes on 
		y_train = predict(w, t_train,model,degree)
		y_test = predict(w, t_test,model,degree)

		#Calculating Mean Square (MS)
		MStrain = calculateMS(y_train, c_train)
		MStest = calculateMS(y_test,c_test)

		print "Mean Square Error for the train set: ",MStrain
		print "Mean Square Error for the test set: ",MStest

		return MStrain,MStest

	""" Bayesian LR - MAIN """
	if method=="bayes":
		bys_MS = np.array([])
		bys_MS1 = [0,sys.maxint]
		bys_MS2 = [0,sys.maxint]

		for alpha in alphas:
			bysMean, bysCovariance = computeBayesianMeanAndCovariance(t_train, c_train, alpha)
			bys_y_train = predict(bysMean, t_train, model, degree)
			bys_y_test = predict(bysMean, t_test, model, degree)

			#calculate Mean Squared (MS) for each variable selection
			MS_train = calculateMS(c_train, bys_y_train)
			MS_test = calculateMS(c_test, bys_y_test)

			#We only want to return the lowest 
			if MS_train < bys_MS1[1]:
				bys_MS1 = [alpha,MS_train]
			if MS_test < bys_MS2[1]:
				bys_MS2 =[alpha,MS_test]

	print "Alpha value and Mean Squared Error for the train set: ",bys_MS1
	print "Alpha value and Mean Square Error for the test set: ",bys_MS2

	return bys_MS1,bys_MS2
