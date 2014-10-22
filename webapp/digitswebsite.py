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
	


	#the following will pull our 1 key out of this weird data structure
	rawData = ""
	#print 'input keys: ', len(inputdata.keys())    #DEBUG
	#							We expect just 1 key

	for key in inputdata.keys():
		rawData = inputdata[key]

	#rawData is now the same Base64 encoded string we had on the client
	
	imageArray = digitClassifier.convertBase64ToImageArray(rawData)
	scaledImageArray = digitClassifier.scaleImageTo20px(imageArray)

	predictionObject = digitClassifier.predictDigit(knn, scaledImageArray)
	# prediction = "Pass through successful"

	return render_template('results2.html', 
		predictedValue=predictionObject[0], 
		predictedProbs=predictionObject[1])



if __name__ == '__main__':
	app.debug = True
	app.run()
