import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
@app.route('/index')
def index():
    '''Home page'''
    return render_template('salary_predict.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    '''Predict the form post value and find the result'''
    if request.method == 'POST':
        #Make the feature is 2 dimensional array
        final_features = [[(int(request.form['years']))]]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
    return render_template('salary_predict.html', result='Employee Salary should be Rs {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)     
