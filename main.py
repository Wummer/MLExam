import numpy as np
import pylab as plt
import PCA
import NORM
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import k_means

""" Some Pylab settings to make the plots look nice with LaTeX.
 If you do NOT have LaTeX then uncomment the following statements """

plt.rc('text', usetex=True)
plt.rc('font', family='Computer Modern',size=16)

"""
------------------------------- Exam Assignment -------------------------------
"""


"""
-------------------------------------------------------------------------------
 Question 1: Linear Regression
-------------------------------------------------------------------------------
"""
print "*"*45,"\n Question 1 \n"

SSTrain = np.loadtxt("data/SSFRTrain2014.dt",unpack=False,delimiter=" ")
SSTest = np.loadtxt("data/SSFRTest2014.dt",unpack=False,delimiter=" ")

print SSTrain[0]





"""
-------------------------------------------------------------------------------
 Question 4: Principal Components Analysis
-------------------------------------------------------------------------------

For comments on the specific functions, i.e. the source code, see
PCA.py & NORM.py 

"""
print "*"*45,"\n Question 4 \n"

SGdata = np.loadtxt("data/SGTrain2014.dt",unpack=False, delimiter=',')
SGTest = np.loadtxt("data/SGTest2014.dt",unpack=False, delimiter=",")

"""
Since we're only interested in dimensionality reduction of one class
 I extract the galaxies and remove their labels.
"""
SGTrain = []
for d in SGdata:
	if d[-1] == 0:
		SGTrain.append(d[:-1])


SGMean = PCA.MLmean(SGTrain)
SGCov = PCA.MLcov(SGTrain,SGMean)
eigw,eigv =  np.linalg.eig(SGCov)
SGNorm = SGTrain#NORM.meanfree(SGTrain)


""" Python doesn't return an ordered list of eigenvalues/eigenvectors 
	so we join them and sort them in descending order.
	Then we substract the 2 highest eigenvectors/principal components """
SGVectors = []
for i in range(len(eigw)):
	SGVectors.append((eigw[i],eigv[:,i]))
SGVectors = sorted(SGVectors, reverse=True, key=lambda tup: tup[0])
SGPC = [SGVectors[0][1],SGVectors[1][1]]

#Projection via dot product
new_SGX,new_SGY = PCA.transform(SGNorm,SGPC)

# Plotting the eigenspectrum
plt.plot(range(1,len(eigw)+1),eigw,'r-')
plt.xlabel('Eigenvector number')
plt.ylabel('Eigenvalue')
plt.title('Eigenspectrum')
plt.show()

#Plotting the projection onto the first 2 Principal Components
plt.plot(new_SGX,new_SGY,"x")
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title("The SGdata projected onto Principal Components")
plt.show()


"""
-------------------------------------------------------------------------------
Question 5: K-means clustering
-------------------------------------------------------------------------------
Here I utilize the SK-learn k_means clustering method and my own PCA library.
"""
print "*"*45,"\n Question 5 \n"

centroids,label,inertia = k_means(SGNorm,n_clusters=2,init="random")

center_x,center_y = PCA.transform(centroids,SGPC)

#plotting the projecting of the centroids onto the 2 PC
plt.plot(new_SGX,new_SGY,"x",label="Projected data")
plt.plot(center_x,center_y,"o",label="K-means centroids")
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.legend(loc="lower left")
plt.title("Projected k-means centroids")
plt.show()


"""
-------------------------------------------------------------------------------
Question 7: Multi-class Classification
-------------------------------------------------------------------------------
In this question I make use of two SK-learn libraries: K-NN and SVM.SVC.

"""
print "*"*45,"\n Question 7 \n\n"

VSTraindata = np.loadtxt("data/VSTrain2014.dt",unpack=False, delimiter=',')
VSTest = np.loadtxt("data/VSTest2014.dt",unpack=False, delimiter=',')
iterations = 10 #Used to control how many runs we want to average over
KNN_totalresult = []
SVM_totalresult = []

K = [1,5,9,13,17,21,25]
VSX = []
VSY = []
new_VSX = []
new_VSY = []

#Extracting the class labels from the data
for d in VSTraindata:
	VSX.append(d[:-1])
	VSY.append(d[-1])
for d in VSTest:
	new_VSX.append(d[:-1])
	new_VSY.append(d[-1])

VSTrain = VSX
VSX = NORM.meanfree(VSTrain)
new_VSX = NORM.transformtest(VSTrain,new_VSX)
print VSTrain[0]
print VSX[0]

"""
Here I run and test the SK-learn KNN with different values of K's
for i number of iterations
"""
for k in K:
	avg_result = []
	print "Training and testing %d-NN"%k
	trainacc = 0
	testacc = 0
	for i in xrange(iterations):
		neigh = KNeighborsClassifier(n_neighbors=k)
		neigh.fit(VSX,VSY)
		trainacc += neigh.score(VSX,VSY)
		testacc += neigh.score(new_VSX,new_VSY)

	avg_result = [trainacc/iterations,testacc/iterations]
	KNN_totalresult.append(avg_result)

""" Restructuring for easy plotting """
trainacc = [a[0] for a in KNN_totalresult]
testacc = [a[1] for a in KNN_totalresult]
plt.plot(K,trainacc,"x-",label="Train acc")
plt.plot(K,testacc,"x-",label="Test acc")
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.title('Accuracies for the K-NN')
plt.show()

print "\n"
"""
Here I run and test the SK-learn implementation of SVM with a linear kernel
 thats based on the LIBSVM. I test multiple values for C and run it 10 times.
 Multi-class strategy is one-by-one.
"""
C = np.linspace(0.00001,1,20)
C2 = np.linspace(0.1,10,10)


for c in C:
	print "Training and testing with C as: ",c
	avg_result = []
	trainacc = 0
	testacc  = 0
	for i in xrange(iterations):
		LinSVM = svm.SVC(C=c,kernel='linear')
		LinSVM.fit(VSX,VSY)
		trainacc += LinSVM.score(VSX,VSY)
		testacc += LinSVM.score(new_VSX,new_VSY)
	avg_result = [trainacc/iterations,testacc/iterations]
	SVM_totalresult.append(avg_result)

""" Restructuring for easy plotting """
trainacc = [a[0] for a in SVM_totalresult]
testacc = [a[1] for a in SVM_totalresult]
plt.plot(C,trainacc,"x-",label="Train acc")
plt.plot(C,testacc,"x-",label="Test acc")
plt.legend(loc='best')
plt.title('Accuracies for the Linear SVM')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.show()

print "\n "

""" Simple test for higher C's """
SVM_totalresult = []
for c in C2:
	print "Training and testing with C as: ",c
	avg_result = []
	trainacc = 0
	testacc  = 0
	for i in xrange(iterations):
		LinSVM = svm.SVC(C=c,kernel='linear')
		LinSVM.fit(VSX,VSY)
		trainacc += LinSVM.score(VSX,VSY)
		testacc += LinSVM.score(new_VSX,new_VSY)
	avg_result = [trainacc/iterations,testacc/iterations]
	SVM_totalresult.append(avg_result)

""" Restructuring for easy plotting """
trainacc = [a[0] for a in SVM_totalresult]
testacc = [a[1] for a in SVM_totalresult]
plt.plot(C2,trainacc,"x-",label="Train acc")
plt.plot(C2,testacc,"x-",label="Test acc")
plt.legend(loc='best')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Accuracies for the Linear SVM with higher C values')
plt.show()