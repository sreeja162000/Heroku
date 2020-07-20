from flask import Flask, request, jsonify, render_template
import pickle
import datetime
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/' , methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/data',methods=['GET','POST'])
def data():
    if request.method=='POST':
        f=request.form['upload-file']
        data=pd.read_excel(f)
        test=data.copy()
        test=test.drop(['Customer ID'],axis=1)
        def convert_date_to_ordinal(date):
            return date.toordinal()
        for i in range(len(test)):
            test['Acquisition date'].iloc[i]=convert_date_to_ordinal(test['Acquisition date'].iloc[i])
            test['Interaction date'].iloc[i]=convert_date_to_ordinal(test['Interaction date'].iloc[i])
        test=test.astype({'Acquisition date':'int64','Interaction date':'int64'})
        test['Category of interaction']=test['Category of interaction'].map({'positive':1,'negative':-1,'neutral':0})
        prediction=model.predict(test)
        data['Churn date']=0
        for i in range(len(data)):
            data['Churn date'][i]=prediction[i]
        data['Churn date']=data['Churn date'].astype(int)
        data['Churn date']=data['Churn date'].map(datetime.datetime.fromordinal)

        data['Churn']='No'
        d=datetime.datetime(2050,12,12)
        for i in range(len(data)):
            if data['Churn date'][i]==d:
                data['Churn'][i]='No'
            else:
                data['Churn'][i]='Yes'

        return render_template('index.html',data=data.to_html())

if __name__ == "__main__":
    app.run(debug=True)

