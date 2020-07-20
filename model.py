import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import datetime
import pickle


data=pd.read_excel('Customer_retention.xlsx')

#DATA MANIPULATION

#grouping data based on customer ID
grouped_data=data.groupby(['Customer ID'])

#acquisition date, interaction date , churn date
#correcting Acquisition dates and churn dates and interaction dates for each customer
columns=['Customer ID','Acquisition date','Interaction date','Category of interaction','Churn date']
df=pd.DataFrame(columns=columns)

d=datetime.datetime(2050,12,12) #date to represent that a person is still a customer

count=0
for i,j in grouped_data:
    
    #acquisition dates
    ad=(j['Acquisition date'].iloc[0]) # first acquistion date of every customer
    j['Acquisition date']=ad
    
    #churn dates
    
    count+=1
    if count%7==0 or count%8==0 or count%9==0:
        j['Churn date']=j['Acquisition date']+pd.DateOffset(days=1200)
    elif count%11==0 or count%17==0 or count%13==0:
        j['Churn date']=j['Acquisition date']+pd.DateOffset(days=1000)
    elif count%10==0 or count%15==0 or count%18==0:
        j['Churn date']=j['Acquisition date']+pd.DateOffset(days=730)
    elif count%23==0 or count%29==0 or count%31==0:
        j['Churn date']=j['Acquisition date']+pd.DateOffset(days=365)
    else:
        j['Churn date']=d #represents the person is still a customer
        
        
    #interaction dates
    for n in range(len(j['Interaction date'])):
        for m in range(n+1,len(j['Interaction date'])):
            if (j['Interaction date'].iloc[n]==j['Interaction date'].iloc[m]): #checks if dates are same within a customer
                j['Interaction date'].iloc[m]=j['Interaction date'].iloc[n]+pd.DateOffset(days=2)

    df=pd.concat([df,j],ignore_index=True) #dataframe after making changes in the dates

#category of interaction
for i in range(len(df)):
    if i%9==0 or i%8==0 or i%11==0:
        df['Category of interaction'][i]='positive'
    elif i%6==0 or i%7==0 or i%10==0:
        df['Category of interaction'][i]='negative'
    else:
        df['Category of interaction'][i]='neutral'

    
#MODELLING

df_=df.copy()
df_=df_.drop(['Customer ID'],axis=1)

def convert_date_to_ordinal(date):
    return date.toordinal()
for i in range(len(df_)):
    
    df_['Acquisition date'].iloc[i]=convert_date_to_ordinal(df_['Acquisition date'].iloc[i])
    df_['Interaction date'].iloc[i]=convert_date_to_ordinal(df_['Interaction date'].iloc[i])
    df_['Churn date'].iloc[i]=convert_date_to_ordinal(df_['Churn date'].iloc[i])

#### converting categorical data to numerical
df_=df_.astype({'Acquisition date':'int64','Interaction date':'int64','Churn date':'int64'})
df_['Category of interaction']=df_['Category of interaction'].map({'positive':1,'negative':-1,'neutral':0})

#### seperating dependent and independent variables
x=df_.drop(['Churn date'],axis=1)
y=df_['Churn date']



#splitting into training and testdata
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,random_state=23,test_size=0.3)


from sklearn.tree import DecisionTreeRegressor as DTR

                         
dtr=DTR(max_depth=41,random_state=23)
dtr.fit(train_x,train_y)



pickle.dump(dtr,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[737000,737002,0]]))

