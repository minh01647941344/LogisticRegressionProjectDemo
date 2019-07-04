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
dataset = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'CV.csv'),encoding='utf-8', errors='ignore') as f:
    dataset.csv = f.read()

@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

def ValuePredictor(to_predict_list,index):
    to_predict = np.array(to_predict_list).reshape(1,7)

    if index == 1:
        loaded_model = pickle.load(open("LogisticRegressionModel.pkl","rb"))
    else:
        loaded_model = pickle.load(open("LogisticRegressionTestModel.pkl","rb"))

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
        result = ValuePredictor(to_predict_list,1)
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
        # Read file and tranform to dataframe
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
        Accuracy = accuracy_score(y_test,predictions)*100
        pickle.dump(logmodel, open("C:\\Users\\HELLO\\Desktop\\Final_Project\\Web_Application\\LogisticRegressionTestModel.pkl","wb"))
        return render_template("TrainedModel.html",Accuracy=Accuracy)

@app.route('/newIndex')
def newIndex():
    return render_template("newIndex.html")

@app.route('/newResult',methods = ['POST'])
def newResult():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list,2)
        if int(result) == 1:
            prediction="đạt"
        else:
            prediction="chưa đạt"
        return render_template("newResult.html",prediction=prediction)


@app.route('/newCompanyCV',methods=['GET'])
def newCompanyCV():
    if request.method == 'GET':
        # Load Model
        loaded_model = pickle.load(open("LogisticRegressionTestModel.pkl","rb"))

        # Declare location of CSV and excel file       
        file_errors_location = 'C:\\Users\\HELLO\\Desktop\\Final_Project\\Web_Application\\Dataset\\CompanyCV.xlsx'
        
        # load excel and csv
        df = pd.read_excel(file_errors_location)
        temp = pd.read_excel(file_errors_location)
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
        resultCSV.to_csv(r'C:\\Users\\HELLO\\Desktop\\Final_Project\\Web_Application\\newCV.csv', index=False)
        newDataset = tablib.Dataset()
        with open(os.path.join(os.path.dirname(__file__),'newCV.csv'),encoding='utf-8', errors='ignore') as f:
            newDataset.csv = f.read()
        data = newDataset.csv

        return render_template("newCompanyCV.html",data=data)

if __name__ == '__main__':
    app.run(debug=True)