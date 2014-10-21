from flask import Flask, render_template, request
from sklearn.externals import joblib
import digitClassifier
import base64

app = Flask(__name__)


print 'Loading model...'
print 'I guess I can do other stuff up here...'

X = digitClassifier.loadInitialData('static/trainingdata.png')

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

	# print 'mydata is: ',inputdata
	# print type(inputdata)
	# print len(inputdata.keys())

	#the following will pull our one key out of this weird data structure
	imageBase64 = ""
	for key in inputdata.keys():
		imageBase64 = key

	#imagebase64 is the raw data over the wire, which is double-encoded.
	imageBase64 = base64.decodestring(imageBase64)
	#now that we've decoded it once, it should work as expected.
	
	digitClassifier.writeImageToFile(imageBase64, 'testoutput.png')
	print
	print
	print

	print digitClassifier.convertBase64ToImageArray(imageBase64)

	return render_template('results.html')



if __name__ == '__main__':
	app.debug = True
	app.run()
