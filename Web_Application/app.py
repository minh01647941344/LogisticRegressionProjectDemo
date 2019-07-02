import os
import numpy as np
import pandas as pd
import flask
import pickle
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import os
import tablib

app = Flask(__name__)
dataset = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'CV.csv'),encoding='utf-8', errors='ignore') as f:
    dataset.csv = f.read()

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,7)
    loaded_model = pickle.load(open("LogisticRegressionModel.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

def changePredictValue(list_predict):
    if list_predict == 1:
        return "Đạt"
    if list_predict == 0:
        return "Chưa đạt"

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction="đạt"
        else:
            prediction="chưa đạt"
        return render_template("result.html",prediction=prediction)

@app.route('/companyCV',methods=['GET'])
def companyCV():
    if request.method == 'GET':
        
        # Declare location of CSV and excel file
        file_csv = 'C:\\Users\\HELLO\\Desktop\\Final_Project\\Web_Application\\CV.csv'
        file_errors_location = 'C:\\Users\\HELLO\\Desktop\\Final_Project\\Web_Application\\Dataset\\CompanyCV.xlsx'

        # load excel and csv
        dfCSV = pd.read_csv(file_csv)
        df = pd.read_excel(file_errors_location)
        
        # Concat Result Column into file Excel
        df = pd.concat([df,dfCSV['Result']],axis=1)

        # Declare Variable contain column
        FullName = df['Full Name']
        Birthday = df['Birthday']
        Address = df['Address']
        Phone = df['Phone']
        Position = df['Position']
        Experiences = df['Year of experiences']
        Education = df['Education']
        Certificates = df['Number of certificates']
        Projects = df['Number of Projects']
        GPA = df['GPA']
        TOIE = df['TOEIC/IELTS']
        Result = df['Result']

        # Convert dataframe to CSV
        dict = {'Full Name':FullName,'Birthday':Birthday,'Address':Address,'Phone':Phone,'Position': Position, 'Year of experiences': Experiences, 'Education': Education, 'Number of certificates':Certificates, 'Number of Projects':Projects, 'GPA':GPA, 'TOEIC/IELTS':TOIE, 'Result':Result} 
        companyCV = pd.DataFrame(dict)
        companyCV.to_csv(r'C:\\Users\\HELLO\\Desktop\\Final_Project\\Web_Application\\CV.csv', index=False)

        #Load file CSV and pass for companyCV.html
        data = dataset.csv
        return render_template("companyCV.html",data=data)


if __name__ == '__main__':
    app.run(debug=True)