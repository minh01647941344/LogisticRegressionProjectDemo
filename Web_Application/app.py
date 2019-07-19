import os
import numpy as np
import pandas as pd
import flask
import pickle
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
import io
import tablib
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,7)
    loaded_model = pickle.load(open("LogisticRegressionModel.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

def markLabel_Position(cols):
    Position = cols
    if Position == 'Designer':
        return 0
    if Position == 'Lập trình viên Frontend Website':
        return 1
    if Position == 'Lập trình viên Backend Website':
        return 2
    if Position == 'Lập trình viên ứng dụng di động':
        return 3
    if Position == 'IT Helpdesk':
        return 4
    if Position == 'IT Support':
        return 5
    if Position == 'Kĩ thuật viên bảo trì':
        return 6
    if Position == 'Kĩ thuật viên hạ tầng mạng':
        return 7
    if Position == 'Kĩ thuật viên an ninh mạng':
        return 8
    if Position == 'Kĩ thuật viên an toàn thông tin':
        return 9
    if Position == 'Chuyên viên quản trị cơ sở dữ liệu':
        return 10
    if Position == 'Chuyên viên phân tích dữ liệu':
        return 11
    if Position == 'Chuyên viên phân tích hệ thống':
        return 12
    if Position == 'Chuyên viên hỗ trợ người dùng':
        return 13
    if Position == 'Chuyên viên quản lý dự án':
        return 14
    if Position == 'Chuyên viên triển khai dự án':
        return 15
    if Position == 'Kĩ thuật viên vận hành thiết bị':
        return 16
    if Position == 'Kĩ thuật viên kiểm thử phần mềm':
        return 17
    if Position == 'Chuyên viên SEO':
        return 18
    if Position == 'Nhân viên Marketing Online':
        return 19
    if Position == 'Nhân viên Marketing Media':
        return 20
    if Position == 'Nhân viên Content Marketing':
        return 21
    if Position == 'Chuyên viên chăm sóc khách hàng':
        return 22
    if Position == 'Nhân viên bán hàng':
        return 23
    if Position == 'Nhân viên tư vấn dịch vụ':
        return 24
    if Position == 'Nhân viên văn phòng':
        return 25
    if Position == 'Nhân viên tổng đài dịch vụ':
        return 26
    if Position == 'Nhân viên lễ tân':
        return 27
    if Position == 'Biên dịch viên':
        return 28
    if Position == 'Nhân viên nhân sự':
        return 29

def markLabel_Education(cols):
    Education = cols
    if Education == 'Đại học Bách Khoa':
        return 0
    if Education == 'Đại học Công nghiệp':
        return 1
    if Education == 'Đại học Công nghệ thông tin':
        return 2
    if Education == 'Đại học Khoa học Tự nhiên':
        return 3
    if Education == 'Đại học GTVT':
        return 4
    if Education == 'Đại học Mở':
        return 5
    if Education == 'Đại học HUTECH':
        return 6
    if Education == 'Đại học ngoại ngữ - tin học':
        return 7
    if Education == 'Đại học sư phạm kĩ thuật':
        return 8
    if Education == 'Đại học Ngoại thương':
        return 9
    if Education == 'Đại học Nông lâm':
        return 10
    if Education == 'Đại học Hoa sen':
        return 11
    if Education == 'Đại học Hồng Bàng':
        return 12
    if Education == 'Đại học Tôn Đức Thắng':
        return 13
    if Education == 'Đại học Trần Đại Nghĩa':
        return 14
    if Education == 'Đại học Văn Lang':
        return 15
    if Education == 'Đại học Nguyễn Tất Thành':
        return 16
    if Education == 'Đại học Văn Hiến':
        return 17
    if Education == 'Đại học Quốc tế Sài Gòn':
        return 18
    if Education == 'Đại học Đồng Nai':
        return 19

def changePredictValue(list_predict):
    if list_predict == 1:
        return "Đậu"
    if list_predict == 0:
        return "Rớt"

@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction="đậu"
        else:
            prediction="rớt"
        return render_template("result.html",prediction=prediction)

@app.route('/trainingAnotherModel')
def trainingAnotherModel():
    return render_template("createNewModel.html")

@app.route('/createModel',methods=['POST'])
def createModel():
    #Check method
    if request.method == 'POST':
        f = request.files['data']
        # Check file exist
        if not f:
            return "No file"
        # Read file and tranform to dataframe and training model
        df = pd.read_excel(f)
        EnglishCertificate = pd.get_dummies(df['TOEIC/IELTS'])
        InterviewResult = pd.get_dummies(df['Interview Result'],drop_first=True)
        df = pd.concat([df,EnglishCertificate,InterviewResult],axis=1)
        df['Position']=df['Position'].apply(markLabel_Position)
        df['Education']=df['Education'].apply(markLabel_Education)
        df.drop(['Full Name','Birthday','Address','Phone','không','TOEIC/IELTS','Interview Result'],axis=1,inplace=True)
        X = df.drop('Pass',axis=1)
        y = df['Pass']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
        logmodel = LogisticRegression(solver='lbfgs',multi_class='auto')
        logmodel.fit(X_train,y_train)
        predictions = logmodel.predict(X_test)
        # evaluate accuracy of model
        Accuracy = accuracy_score(y_test,predictions)*100
        # pack model
        pickle.dump(logmodel, open("LogisticRegressionModel.pkl","wb"))
        
        return render_template("TrainedModel.html",Accuracy=Accuracy)

@app.route('/selectedCompanyCV')
def selectedCompanyCV():
    return flask.render_template('selectedCompanyCV.html')

@app.route('/companyCV',methods=['POST'])
def companyCV():
    if request.method == 'POST':
        # Load Model
        loaded_model = pickle.load(open("LogisticRegressionModel.pkl","rb"))

        # get response from selectedCompanyCV
        f = request.files['data']
        if not f:
            return "No file"
        
        # load excel and csv
        df = pd.read_excel(f)
        temp = pd.read_excel(f)
        
        # Prepare data and clean data
        EnglishCertificate = pd.get_dummies(df['TOEIC/IELTS'])
        df = pd.concat([df,EnglishCertificate],axis=1)
        df['Position']=df['Position'].apply(markLabel_Position)
        df['Education']=df['Education'].apply(markLabel_Education)
        df.drop(['Full Name','Birthday','Address','Phone','không','TOEIC/IELTS'],axis=1,inplace=True)

        # Predict data
        prediction = loaded_model.predict(df)
        dataframeOfPredict = pd.DataFrame(prediction, columns=['Result'])
        dataframeOfPredict['Result'] = dataframeOfPredict['Result'].apply(changePredictValue)

        # Result
        resultCSV = pd.concat([temp,dataframeOfPredict],axis=1)

        # Convert to CSV and read file
        resultCSV.to_csv(r'CV.csv', index=False)
        newDataset = tablib.Dataset()
        with open(os.path.join(os.path.dirname(__file__),'CV.csv'),encoding='utf-8', errors='ignore') as f:
            newDataset.csv = f.read()
        data = newDataset.csv

        return render_template("companyCV.html",data=data)

if __name__ == '__main__':
    app.run(debug=True)