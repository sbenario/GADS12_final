
from scipy import ndimage, misc
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def loadInitialData(datafile):
	"""Input: string representing path to data file. Output is numpy array with data."""

	digitsdata = ndimage.imread(datafile, flatten=True, mode='L')

	cells = [np.hsplit(row,100) for row in np.vsplit(digitsdata,50)]
	# There should be 5000 individiual cells here.
	# We just split this into 50 rows, and 100 columns per row.

	x = np.array(cells)
	#x.shape =  (50, 100, 20, 20)

	X = x.reshape(5000,20,20)
	X2 = np.reshape(X, (len(X), 400))
	#400 comes out of the 20x20 shape of each entry.

	#return the loaded data as a long array.
	#each row is 400 features wide representing each 20x20 pixel
	return X2


def trainClassifier(X):
	"""Input: Numpy array representing X features. Output is a trained classifier."""

	knn = KNeighborsClassifier(algorithm='auto', leaf_size=13, metric='minkowski',
           n_neighbors=5, p=2, weights='uniform')
	
	# Create labels for the data. These are the labels for our MNIST data
	y = np.repeat(np.arange(10),500)

	knn.fit(X, y)
	return knn


# def predcitDigit(digitArray):


