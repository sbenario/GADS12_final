
from scipy import ndimage, misc
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize as imageresize
import base64


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

def convertBase64ToImageArray(rawdata):
    rawdata = rawdata+"=="
    trimmed = rawdata[22:]   #we need to trim the begining of the blob
    print 'trimmed is: ', trimmed
    decodedString = base64.decodestring(trimmed)
    
    #May god have mercy on my soul for this horrible hack
    tempfile = 'tempfile.png'
    f = open(tempfile, 'w')
    f.write(decodedString)
    f.close()
    theActualImage = ndimage.imread(tempfile)

    theActualImage = theActualImage[:,:,3]  #we only want the last of the 4 dimensions

    #we don't shrink this yet so that it can still be accessed for debugging purposes
    return theActualImage


def scaleImageTo20px(arrayOfImage):
    smallimage = imageresize(arrayOfImage, (20,20))
#     smallimage = smallimage.reshape(400,)
	#image is being returned in a 20x20 array. Needs to be reshaped still.
    return smallimage

def writeImageToFile(rawimage, filename):
	"""writeImageToFile(rawimage, filename) -- both strings"""
	# filename = 'thefile.png'
	decodedString = base64.decodestring(rawimage[22:])

	f = open(filename, 'w')
	f.write(decodedString)
	f.close()
	return

def predictDigit(clf, digitArray):
	"""input is a 20x20 array of a small image. Output is tuple(predicted value, array of probabilities)"""
	features = digitArray.reshape(400,)
	return clf.predict(features)

def decodeBadPaddingBase64(string64):
	"""Given a base64 encoded string which is missing some number of = on the end, add them and decode the string"""
	for stringoptions in [string64, string64+"=", string64+"=="]:
		try:
			return base64.decodestring(stringoptions)
		except:
			pass





