import numpy as np
import pylab as plt
import PCA
import NORM
import REGRESSION as REG
import CROSSVAL
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.cluster import k_means

np.random.seed(10)

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

For comments on the specific functions, i.e. the source code, see
REGRESSION.py & NORM.py 

"""
print "*"*75,"\n Question 1 \n"

SSTrain = np.loadtxt("data/SSFRTrain2014.dt",unpack=False,delimiter=" ")
SSTest = np.loadtxt("data/SSFRTest2014.dt",unpack=False,delimiter=" ")

np.random.shuffle(SSTrain)
np.random.shuffle(SSTest)

#Getting the vectors of all target variables
SSX_train = SSTrain[:,:-1]
SSX_test = SSTest[:,:-1]

#Extracting the class information
SSY_train = SSTrain[:,-1]
SSY_test = SSTest[:,-1]


print "Maximum Likelihood Regression: \n"
ML_MStrain, ML_MStest = REG.run(SSX_train,SSY_train,SSX_test,SSY_test,
	model="linear")

print "\n"
"""
-------------------------------------------------------------------------------
 Question 2: Non-Linear Regression
-------------------------------------------------------------------------------

For comments on the specific functions, see REGRESSION.py

"""
print "*"*75,"\n Question 2 \n"
#Inherit the alpha values from above
degrees = np.arange(2,5,1)
alphas = np.arange(0.,30.,1)

print "Finding optimal non-linear regression parameters."
print  "Please wait while it calculates..."
bestdeg,bestalpha = REG.MAP_Gridsearch(
	SSX_train,SSY_train,SSX_test,SSY_test,alphas,degrees, model="poly")


print "\nMaximum A posteriori Regression"
REG.run(SSX_train,SSY_train,SSX_test,SSY_test,
	method="map",model="poly", degree=bestdeg,alpha=bestalpha)



"""
-------------------------------------------------------------------------------
 Question 3:Binary Classification using Support Vector Machines
-------------------------------------------------------------------------------

For comments on the specific functions, i.e. the source code, see
 CROSVALL.py.

"""
print "*"*75,"\n Question 3 \n"

SGdata = np.loadtxt("data/SGTrain2014.dt",unpack=False, delimiter=',')
SGTest = np.loadtxt("data/SGTest2014.dt",unpack=False, delimiter=",")

np.random.shuffle(SGdata)
np.random.shuffle(SGTest)

SGX = []
SGY = []
new_SGX = []
new_SGY = []

#Extracting the class labels from the data
for d in SGdata:
	SGX.append(d[:-1])
	SGY.append(d[-1])
for d in SGTest:
	new_SGX.append(d[:-1])
	new_SGY.append(d[-1])

SGTrain = SGX
SGX = NORM.meanfree(SGTrain)
new_SGX = NORM.transformtest(SGTrain,new_SGX)

print "Finding optimal Gamma and C pair parameters."
print  "Please wait while it calculates...\n"
c,g = CROSSVAL.SVM_Gridsearch(SGX, SGY, 5)

SVC = svm.SVC(kernel="rbf",gamma=g,C=c)
SVC.fit(SGX,SGY)

print "RBF SVM Loss on train: ", 1-SVC.score(SGX,SGY)
print "RBF SVM Loss on test: ", 1-SVC.score(new_SGX,new_SGY)


print "\n"
"""
-------------------------------------------------------------------------------
 Question 4: Principal Components Analysis
-------------------------------------------------------------------------------

For comments on the specific functions, i.e. the source code, see
PCA.py & NORM.py 

"""
print "*"*75,"\n Question 4 \n"

"""
Since we're only interested in dimensionality reduction of one class
 we extract the galaxies and remove their labels.
"""
SGTrain = []
for d in SGdata:
	if d[-1] == 0:
		SGTrain.append(d[:-1])

#Normalizing the data 
SGNorm = NORM.meanfree(SGTrain)

#Sampling mean and covariance from the normalized distribution
SGMean = PCA.MLmean(SGNorm)
SGCov = PCA.MLcov(SGNorm,SGMean)
eigw,eigv = np.linalg.eig(SGCov)


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
print "*"*75,"\n Question 5 \n"

centroids,label,inertia = k_means(SGNorm,n_clusters=2,init="random")

center_x,center_y = PCA.transform(centroids,SGPC)

#plotting the projecting of the centroids onto the 2 PC
plt.plot(new_SGX,new_SGY,"x",label="Projected data")
plt.plot(center_x,center_y,"o",label="K-means centroids")
plt.xlabel('First principal component')	
plt.ylabel('Second principal component')
plt.legend(loc="best")
plt.title("Projected k-means centroids")
plt.show()


"""
-------------------------------------------------------------------------------
Question 7: Multi-class Classification
-------------------------------------------------------------------------------
In this question I make use of two SK-learn libraries: K-NN and SVM.SVC.

"""
print "*"*75,"\n Question 7 \n\n"

VSTraindata = np.loadtxt("data/VSTrain2014.dt",unpack=False, delimiter=',')
VSTest = np.loadtxt("data/VSTest2014.dt",unpack=False, delimiter=',')
np.random.shuffle(VSTraindata)
np.random.shuffle(VSTest)

iterations = 10 #Used to control how many runs we want to average over
KNN_totalresult = []
SVM_totalresult = []

""" Parameters for the classifiers"""
K = np.arange(1,15,1)
C= np.arange(0.005,1,0.005)

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


"""Testing for the best K-setting via cross-validation"""
print "Finding the optimal K value."
print "Please wait while it calculates..."
Best_K = CROSSVAL.KNN_Crossvalidation(VSX,VSY, 5, K)

KNN = KNeighborsClassifier(n_neighbors=Best_K)
KNN.fit(VSX,VSY)
print "KNN Loss on train: ",1-KNN.score(VSX,VSY)
print "KNN Loss on test: ",1-KNN.score(new_VSX,new_VSY)

print "\n"

"""
Here I run and test the SK-learn implementation of SVM with a linear kernel
 thats based on the LIBSVM. I test multiple values for C via cross-validation.
 Then I run the classifier on the test set 10 times and average over the results.
 Multi-class strategy is one-by-one.
"""

print "Finding the optimal C value."
print "Please wait while it calculates..."
Best_C = CROSSVAL.LinSVM_Crossvalidation(VSX,VSY,5,C)

LinSVM = svm.SVC(kernel='linear',C=Best_C)
LinSVM.fit(VSX,VSY)

print "Linear SVM Loss on train: ", 1-LinSVM.score(VSX,VSY)
print "Linear SVM Loss on test: ", 1-LinSVM.score(new_VSX,new_VSY)