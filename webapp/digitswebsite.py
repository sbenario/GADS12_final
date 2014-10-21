from flask import Flask, render_template, request
from sklearn.externals import joblib
import digitClassifier

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
	input = request.form
	return render_template('results.html')



if __name__ == '__main__':
	app.debug = True
	app.run()
