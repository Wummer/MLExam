from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
from operator import itemgetter

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
	return feature_slices,label_slices



"""
This function expects a 1D list of training samples and a 1D list of their labels.
It then computes the Jaakkola parameters with the given formula.
It returns the gamma Jakkola
"""
def jaakkola(X_train,y_train):
	G = []
	for i,n in enumerate(y_train):
		temp = []
		for j,k in enumerate(y_train):
			if n!=k:
				D = np.sqrt((np.sum(X_train[i]-X_train[j]))**2)
				temp.append(D)

		#Ascending order	
		temp = sorted(temp)
		G.append(temp[0])
	sigma = np.median(np.array(G))

	""" Rewriting the Jaakkola equation we get: """
	y_jaakkola = 1/(2*(sigma**2))

	print "The sigma_jaakkola is: ",sigma
	print "The y_jaakkola is: ",y_jaakkola
	return y_jaakkola



"""
The function expects a train set, a 1D list of train labels and number of folds. 
The function has dicts of all C's and gammas. For each combination it runs 5 fold crossvalidation: 
For every test-set for as many folds as there are: use the remaining as train sets (exept if it's the test set.) 
Then we sum up the test and train result for every run and average it. The average performances per combination is stored.
The lowest test average and the combination that produced it is returned with the train error rate.   
"""
def SVM_Gridsearch(X_train, y_train, folds):
	# Set the parameters by cross-validation
	y_jaakkola = jaakkola(X_train,y_train)
	tuned_parameters = [{'gamma': [y_jaakkola * np.exp(-8),y_jaakkola * np.exp(-7), y_jaakkola * np.exp(-6),y_jaakkola * np.exp(-5),
									y_jaakkola * np.exp(-4),y_jaakkola * np.exp(-3), y_jaakkola*np.exp(-2),
									y_jaakkola*np.exp(-1), y_jaakkola*np.exp(0), y_jaakkola*np.exp(1),
									y_jaakkola*np.exp(2), y_jaakkola*np.exp(3)],
                     	'C': [np.exp(-2),np.exp(-1),1,np.exp(1),np.exp(2),np.exp(3)]}]

	features_slices, labels_slices = sfold(X_train, y_train, folds)
	accuracy = []

	""" Gridsearch """
	for g in tuned_parameters[0]['gamma']:
		for c in tuned_parameters[0]['C']:
			temp_train = 0
			temp_test = 0
			
			""" Crossvalidation """
			for f in xrange(folds):
				crossvaltrain = []
				crossvaltrain_labels = []

				crossvaltest = np.array(features_slices[f])
				crossvaltest_labels = np.array(labels_slices[f])
				
				for i in xrange(folds):
					if i != f: 
						for elem in features_slices[i]:
							crossvaltrain.append(elem)
							
						for lab in labels_slices[i]:
							crossvaltrain_labels.append(lab)
				
				crossvaltrain = np.array(crossvaltrain)
				crossvaltrain_labels = np.array(crossvaltrain_labels)

			#Training the classifier
			SVC= svm.SVC(kernel="rbf",C=c,gamma=g)
			SVC.fit(crossvaltrain,crossvaltrain_labels)
			acctrain = SVC.score(crossvaltrain,crossvaltrain_labels)
			acctest = SVC.score(crossvaltest,crossvaltest_labels)
			temp_train += acctrain
			temp_test += acctest

			av_result = [c,g,temp_train/folds,temp_test/folds]
			accuracy.append(av_result)			

	#After all C's and gammas have been tried: get the best performance and the hyperparam pairs for that:
	accuracy = sorted(accuracy, reverse=True, key=itemgetter(-1)) #sort by accuracy!- i.e. highest first
	print accuracy[:5]
	besttrain = accuracy[0][-2]
	besttest = accuracy[0][-1]
	bestpair = tuple(accuracy[0][:2])
	print "\nBest hyperparameter (C, gamma)", bestpair
	return bestpair



"""
The function expects a train set, a 1D list of train labels and number of folds, and a list of C values. 
For each value of C it runs 5 fold crossvalidation: 
For every test-set for as many folds as there are: use the remaining as train sets (except if it's the test set.) 
Then we sum up the test and train result for every run and average it. The average performances per combination is stored.
The highest test average accuracy and the K value that produced it, is returned along with the train error rate.   
"""

def LinSVM_Crossvalidation(x_train, y_train, folds, C):
	slices,labels = sfold(x_train,y_train,folds)
	result = []
	bestresult = []

	for c in C:
		temp_test = 0
		temp_train = 0

		for f in xrange(folds):
			crossvaltest = slices[f]
			crossvaltest_labels = labels[f]
			crossvaltrain =[]
			crossvaltrain_labels = []

			for i in xrange(folds):
				if i != f:
					for elem in slices[i]:
						crossvaltrain.append(elem)
					for lab in labels[i]:
						crossvaltrain_labels.append(lab)

			crossvaltrain = np.array(crossvaltrain)
			crossvaltrain_labels = np.array(crossvaltrain_labels)

			SVC= svm.SVC(kernel="linear",C=c)
			SVC.fit(crossvaltrain,crossvaltrain_labels)

			acctrain = SVC.score(crossvaltrain,crossvaltrain_labels)
			acctest = SVC.score(crossvaltest,crossvaltest_labels)
			temp_train += acctrain 
			temp_test += acctest

	
		av_result = [c,temp_train/folds,temp_test/folds]
		bestresult.append(av_result)

	bestresult = sorted(bestresult,reverse=True,key=lambda lis: lis[2])
	best_c = bestresult[0][0]
	best_train = 1 - bestresult[0][1]
	best_test = 1 - bestresult[0][2]

	print "Best C: %f, Train Loss = %f, Test Loss= %f " %(best_c,best_train,best_test)
	return best_c


"""
The function expects a train set, a 1D list of train labels and number of folds, and a list of K values. 
For each value of K it runs 5 fold crossvalidation: 
For every test-set for as many folds as there are: use the remaining as train sets (except if it's the test set.) 
Then we sum up the test and train result for every run and average it. The average performances per combination is stored.
The highest test average accuracy (however it is returned as 0-1 loss) and the combination that produced it is returned with the train error rate.   
"""
def KNN_Crossvalidation(x_train, y_train, folds, K):
	slices,labels = sfold(x_train,y_train,folds)
	result = []
	bestresult = []

	for k in K:
		temp_test = 0
		temp_train = 0

		for f in xrange(folds):
			crossvaltest = slices[f]
			crossvaltest_labels = labels[f]
			crossvaltrain =[]
			crossvaltrain_labels = []

			for i in xrange(folds):
				if i != f:
					for elem in slices[i]:
						crossvaltrain.append(elem)
					for lab in labels[i]:
						crossvaltrain_labels.append(lab)

			crossvaltrain = np.array(crossvaltrain)
			crossvaltrain_labels = np.array(crossvaltrain_labels)

			KNN = KNeighborsClassifier(n_neighbors=k)
			KNN.fit(crossvaltrain,crossvaltrain_labels)

			acctrain = KNN.score(crossvaltrain,crossvaltrain_labels)
			acctest = KNN.score(crossvaltest,crossvaltest_labels)
			temp_train += acctrain 
			temp_test += acctest


			#print "Cross eval: number of same %d" %countsame	
		av_result = [k,temp_train/folds,temp_test/folds]
		bestresult.append(av_result)

	bestresult = sorted(bestresult,reverse=True,key=lambda lis: lis[2])
	best_k = bestresult[0][0]
	best_train = 1 - bestresult[0][1]
	best_test = 1- bestresult[0][2]

	print "Best K: %d, Train Loss = %f, Test Loss = %f " %(best_k,best_train,best_test)
	return best_k