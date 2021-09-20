from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import pickle            

# Keras

# Flask utils
from flask import Flask, redirect, url_for, request, render_template


# Define a flask app
app = Flask(__name__)

# Model Path
md_log=pickle.load(open('models/logistic_model.pkl', 'rb'))
word_vectorizer= pickle.load(open('models/word_vectorizer.pkl', 'rb'))
user_predicted_ratings = pd.read_pickle("models/user_predicted_ratings.pkl")
df=pd.read_csv("clean_data.csv")

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_name = str(request.form.get('reviews_username'))
    print(user_name)
    prediction = top_5_recommendation(user_name)
    print("Output :", prediction)

    return render_template('index.html', message='We have picked 5 items that may interest you:\n ', username = user_name,results = prediction)
def top_5_recommendation(user_input):
    arr = user_predicted_ratings.loc[user_input].sort_values(ascending=False)[0:20]

    # Based on positive sentiment percentage.
    i= 0
    a = {}
    for prod_name in arr.index.tolist():
        product = prod_name
        product_name_review_list =df[df['name']== product]['review_final'].tolist()
        features= word_vectorizer.transform(product_name_review_list)
        md_log.predict(features)
        a[product] = md_log.predict(features).mean()*100
    b= pd.Series(a).sort_values(ascending = False).head(5).index.tolist()
    #print("Enter username : ",user_input)
    #print("Five Recommendations for you :")
    result= pd.Series(a).sort_values(ascending = False).head(5).index.tolist()
    return result            
        
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)

