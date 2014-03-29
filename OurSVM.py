from __future__ import division
from sklearn.svm import libsvm
from sklearn.neighbors import KNeighborsClassifier as KNN
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

	#random.shuffle(featurefold, lambda: 0.5) 
	#random.shuffle(labelfold, lambda: 0.5) #using the same shuffle 
	feature_slices = [featurefold[i::s] for i in xrange(s)]
	label_slices = [labelfold[i::s] for i in xrange(s)]
	return label_slices, feature_slices


"""
The function expects a train set, a 1D list of train labels and number of folds. 
The function has dicts of all C's and gammas. For each combination it runs 5 fold crossvalidation: 
For every test-set for as many folds as there are: use the remaining as train sets (exept if it's the test set.) 
Then we sum up the test and train result for every run and average it. The average performances per combination is stored.
The lowest test average and the combination that produced it is returned with the train error rate.   
"""
def Gridsearch(X_train, y_train, folds):
	# Set the parameters by cross-validation
	tuned_parameters = [{'gamma': [0.00000000001, 0.0000000001,0.000000001, 0.00000001, 0.0000001, 0.000001,0.00001,0.0001,0.001,0.01,0.1,1],
                     'C': [0.001,0.01,0.1,1,10,100]}]
	
	labels_slices, features_slices = sfold(X_train, y_train, folds)
	accuracy = []

	#gridsearch
	for g in tuned_parameters[0]['gamma']:
		for c in tuned_parameters[0]['C']:
			temp = []
			tr_temp = []
			#crossvalidation
			for f in xrange(folds):
				crossvaltrain = []
				crossvaltrain_labels = []

				#define test-set for this run
				crossvaltest = np.array(features_slices[f])
				crossvaltest_labels = np.array(labels_slices[f])
				
				#define train set for this run
				for i in xrange(folds): #putting content of remaining slices in the train set 
					if i != f: # - if it is not the test slice: 
						for elem in features_slices[i]:
							crossvaltrain.append(elem) #making a list of trainset for this run
							
						for lab in labels_slices[i]:
							crossvaltrain_labels.append(lab) #...and a list of adjacent labels
				
				crossvaltrain = np.array(crossvaltrain)
				crossvaltrain_labels = np.array(crossvaltrain_labels)

				#Classifying using libsvm
				out = libsvm.fit(crossvaltrain, crossvaltrain_labels, svm_type=0, C=c, gamma=g)
				train_y_pred = libsvm.predict(crossvaltrain, *out)
				y_pred = libsvm.predict(crossvaltest, *out)

				#getting the train error count
				tr_count = 0
				for l in xrange(len(crossvaltrain_labels)):
					if train_y_pred[l] != crossvaltrain_labels[l]:
						tr_count +=1
				tr_temp.append(tr_count / len(crossvaltrain))

				#getting the test error count
				counter = 0
				for y in xrange(len(y_pred)):
					if y_pred[y] != crossvaltest_labels[y]:
						counter +=1
				#and storing the error result. 
				temp.append(counter / len(crossvaltest))

			#for every setting, get the average performance of the 5 runs:
			trainmean = np.array(np.mean(tr_temp))
			testmean = np.array(np.mean(temp))
			print "Average test error of %s: %.6f" %((c,g), testmean)
			accuracy.append([c,g,testmean, trainmean])

	#After all C's and gammas have been tried: get the best performance and the hyperparam pairs for that:
	accuracy.sort(key=itemgetter(2)) #sort by error - lowest first
	bestperf = accuracy[0][-2]
	besttrain = accuracy[0][-1]
	bestpair = tuple(accuracy[0][:2])
	print "\nBest hyperparameter (C, gamma)", bestpair
	return bestpair


def LinSVM_Crossvalidation(X_train, y_train, folds):
	"derp"



def KNN_Crossvalidation(x_train, y_train, folds, K):
	slices,labels = sfold(x_train,y_train,folds)
	result = []

	for k in K:
		temp_test = 0
		temp_train = 0

		for f in xrange(folds):
			countsame = 0
			crossvaltest = slices[f]
			crossvaltrain =[]
			
			for i in xrange(folds):
				if i != f:
					for elem in slices[i]:
						crossvaltrain.append(elem)

			KNN = KNN(n_neighbor=k)
			KNN.fit(crossvaltrain,crossvaltest)
			acctrain = KNN.score(crossvaltrain,crossvaltrain)
			acctest = KNN.score(crossvaltrain,crossvaltest)
			temp_train += acctrain 
			temp_test += acctest


			#print "Cross eval: number of same %d" %countsame	
		av_result = [k,temp_train/folds,temp_test/folds]

	bestresult = sorted(av_result,reverse=True,key=itemgetter(3))
	best_k = bestresult[0][0]
	best_train = bestresult[0][1]
	best_test = bestresult[0][2]
	print print "Best K: %d, Train Acc = %f, Test Acc = %f " %(best_k,best_train,best_test)
	
	return bestresult[0]