from flask import Flask, render_template, request
from sklearn.externals import joblib
import digitClassifier
import base64
import StringIO

app = Flask(__name__)


print 'Loading model...'
print 'I guess I can do other stuff up here...'

X = digitClassifier.loadInitialData('static/trainingdata.png')
knn = digitClassifier.trainClassifier(X)



# import views

# @app.route('/')
# def display_form():
#     return render_template('lyrics_form.html')

@app.route('/')
@app.route('/index')
def index():
    user = {'nickname': 'Miguel'}  # fake user
    return render_template('index.html')

@app.route('/input', methods=['POST'])
def numberInput():
	inputdata =  request.form  #type is unfortunately werkzeug.datastructures.CombinedMultiDict
	
	print 'type of inputdata ', type(inputdata)
	print inputdata

	#the following will pull our one key out of this weird data structure
	rawData = ""
	print 'input keys: ', len(inputdata.keys())
	for key in inputdata.keys():
		rawData = inputdata[key]

	# stringObject = StringIO.StringIO(rawData)
	print
	print
	print
	print 'my rawdata:', rawData  #rawdata should now be a binary blob
	print
	print
	print len(rawData)
	print type(rawData)    #RAWDATA IS NOW THE SAME BASE64 STRING AS ON THE CLIENT!!!
	t = open('binarypng.png', 'wb')
	t.write(rawData)
	t.close()
	print
	print base64.encodestring(rawData)
	print
	imageArray = digitClassifier.convertBase64ToImageArray(rawData)

	# #rawData is the raw data over the wire, which is double-encoded.
	# # print str(rawData)
	# # print type(rawData)
	# print

	# #we have discovered that often times the data comes back with bad amounts of padding	
	# # decodedOnce = digitClassifier.decodeBadPaddingBase64(rawData)

	# #now that we've decoded it once, it should work as expected.

	# # digitClassifier.writeImageToFile(imageBase64, 'testoutput.png')
	# print
	# print
	# print

	# imageArray = digitClassifier.convertBase64ToImageArray(rawData)
	# scaledImageArray = digitClassifier.scaleImageTo20px(imageArray)

	# prediction = digitClassifier.predictDigit(knn, scaledImageArray)
	prediction = "Pass through successful"

	return render_template('results.html', result=prediction )



if __name__ == '__main__':
	app.debug = True
	app.run()
