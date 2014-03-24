import numpy as np
import pylab as plt
import PCA
from sklearn.decomposition import PCA as SKPCA

""" Some Pylab setting to make it look nice with LaTeX. If you do NOT have LaTeX then uncomment the following statements """
plt.rc('text', usetex=True)
plt.rc('font', family='Computer Modern',size=16)

"""
----------------------------------------------------------------------------------------------------------------------------
"""

SGdata = np.loadtxt("data/SGTrain2014.dt",unpack=False, delimiter=',')
SGTest = np.loadtxt("data/SGTest2014.dt",unpack=False, delimiter=",")

SGTrain = []
for d in SGdata:
	if d[-1] == 0:
		SGTrain.append(d)

""" Question 4 

For comments on the specific  source code see PCA.py.

"""
SGMean = PCA.MLmean(SGTrain)
SGCov = PCA.MLcov(SGTrain,SGMean)
eigw,eigv =  np.linalg.eig(SGCov)
SGNorm = PCA.normalize(SGTrain,SGMean)

SGVectors = []


""" Python doesn't return an ordered list of eigenvalues/eigenvectors 
	so we join them and sort them in descending order """
for i in range(len(eigw)):
	SGVectors.append((eigw[i],eigv[:,i]))
SGVectors = sorted(SGVectors, reverse=True, key=lambda tup: tup[0])

SGPC = [SGVectors[0][1],SGVectors[1][1]]

new_x,new_y = PCA.transform(SGNorm,SGPC)

# Plotting the eigenspectrum
plt.plot(range(1,len(eigw)+1),eigw,'r-')
plt.xlabel('Eigenvector number')
plt.ylabel('Eigenvalue')
plt.title('Eigenspectrum')
plt.show()

#Plotting the projection onto the first 2 Principal Components

plt.plot(new_x,new_y,"x")
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.show()


"""
Question 6: K-means clustering
"""


"""
Question 7: Multi-class Classification
"""

