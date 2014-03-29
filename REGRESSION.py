from __future__ import division
from operator import itemgetter
import numpy as np
import pylab as plt
import sys

np.random.seed(1)
""" REGRESSION """

"""
Create the design matrix from vector X - either for linear or polynomial basis functions
- Linear design matrices would look like this: [1, x1, ..., xn]
- Polynomial design matrices would look like this: (e.g. with 3 variables)
[1, x1, x2, x3, x1^n, x2^n, x3^n, x1*x2, x1*x3, x2*x3] for each row 
"""
def createDesignMatrix(X, model="linear",degree=2):
	l = len(X[0])
	size = 1 + l

	if model=="poly":
		size += l + (((l-1)*l) / 2)

	phi = np.zeros((len(X), size))

	#set first column values to 1
	phi[:,0] = 1 

	for c in range(l):
		phi[:,c+1] = X[:,c] # phi(x) = x

		if model=="poly":
			phi[:,c+1+l] = X[:,c] ** degree # phi(x) = x**n

	if model=="poly":
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
	phi = createDesignMatrix(X_test,model,degree)
	y = np.zeros(len(phi))

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
def computeBayesianMeanAndCovariance(X, t, degree, alpha,model):
	beta = 1
	#get design matrix
	phi = createDesignMatrix(X,model,degree)

	#Creating the covariance matrix
	bpp = beta * np.dot(phi.T, phi)
	aI = np.zeros(bpp.shape)
	np.fill_diagonal(aI, alpha)
	covariance = aI + bpp
 	
 	#get each part of the mean equation
 	bs = beta * np.linalg.pinv(covariance)
 	mean = np.dot(bs, np.dot(phi.T, t))
 	mean = mean.reshape(-1,len(mean))[0]

 	return mean, covariance  


""" 
MAIN
"""
def run(t_train,c_train,t_test,c_test,method="ml",model="linear",degree=2,alpha=1):

	if method=="ml":
		#Building design matrix
		phi_train = createDesignMatrix(t_train,model,degree)
		phi_test = createDesignMatrix(t_test,model,degree)

		#Getting weight vectors
		w = findML(phi_train, c_train)

		#Getting the predicted target classes on 
		y_train = predict(w, t_train,model,degree)
		y_test = predict(w, t_test,model,degree)

		#Calculating Mean Squared Error (MS)
		MStrain = calculateMS(y_train, c_train)
		MStest = calculateMS(y_test,c_test)

		print "Mean Squared Error for the train set: ",MStrain
		print "Mean Squared Error for the test set: ",MStest

		return MStrain,MStest

	""" Bayesian LR - MAIN """
	if method=="map":
		bys_MS1 = [0,sys.maxint]
		bys_MS2 = [0,sys.maxint]

		bysMean, bysCovariance = computeBayesianMeanAndCovariance(t_train, c_train, degree, alpha, model)
		bys_y_train = predict(bysMean, t_train, model, degree)
		bys_y_test = predict(bysMean, t_test, model, degree)

		#calculate Mean Squared (MS) for each variable selection
		MS_train = calculateMS(c_train, bys_y_train)
		MS_test = calculateMS(c_test, bys_y_test)


	print "Mean Squared Error for the train set: ",MS_train
	print "Mean Squared Error for the test set: ",MS_test

	return MS_train,MS_test



"""
This function splits the shuffled train set in s equal sized splits. 
It expects the features, the labels and number of slices. 
It starts by making a copy of the labels and features and shuffles them. The lambda constant makes sure that it's always shuffled the same way 
It returns a list of s slices containg lists of datapoints belonging to s.
"""
def sfold(features, labels, s):
	featurefold = np.copy(features)
	labelfold = np.copy(labels)

	feature_slices = [featurefold[i::s] for i in xrange(s)]
	label_slices = [labelfold[i::s] for i in xrange(s)]
	return label_slices, feature_slices


def MAP_Gridsearch(t_train, c_train, t_test, c_test ,alphas,degrees,model="map"):
	folds = 5
	results = []
	labels_slices, features_slices = sfold(t_train, c_train, folds)

	for deg in degrees: 
		for alpha in alphas:
			te_temp = 0
			tr_temp = 0

			#crossvalidation
			for f in xrange(folds):
				cv_train = []
				cv_labels = []

				#define test-set for this run
				cv_test = np.array(features_slices[f])
				cv_test_labels = np.array(labels_slices[f])
				
				#define train set for this run
				for i in xrange(folds):  
					if i != f:
						for elem in features_slices[i]:
							cv_train.append(elem)
							
						for lab in labels_slices[i]:
							cv_labels.append(lab) #...and a list of adjacent labels
			
				cv_train = np.array(cv_train)
				cv_labels = np.array(cv_labels)

				""" 
				We then run the train/test splits with the MAP Regression
				"""
				Mean, Covariance = computeBayesianMeanAndCovariance(cv_train, cv_labels, deg, alpha, model)
				y_pred_tr = predict(Mean, cv_train, "poly",deg)
				y_pred = predict(Mean, cv_test, "poly",deg)
					
				MSE_tr = calculateMS(y_pred_tr, cv_labels)
				MSE_te = calculateMS(y_pred, cv_test_labels)
				
				tr_temp += MSE_tr
				te_temp += MSE_te

			tr_temp = tr_temp / folds
			te_temp = te_temp / folds
			results.append([tr_temp, te_temp, (deg, alpha)])


	results = sorted(results,reverse=False, key=itemgetter(1) )
	bestdeg_alpha = results[0][-1]
	train_MSE = results[0][0]
	test_MSE = results[0][1]

	print "Best (degree, alpha): %s, train MSE = %.6f, test MSE = %.6f " %(bestdeg_alpha, train_MSE, test_MSE)
	return bestdeg_alpha