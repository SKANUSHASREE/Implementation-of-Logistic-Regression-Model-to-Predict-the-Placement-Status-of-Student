# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, drop unnecessary columns, and encode categorical variables.
2. Define the features (X) and target variable (y).
3. Split the data into training and testing sets.
4. Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: S KANUSHA SREE
RegisterNumber:  212224040149

import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no", "salary"], axis=1) 
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"]) 
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"]) 
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression (solver="liblinear") 
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred) 
confusion 
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,0,1,1,90,1,0,85,1,85]])
/*
```
## Output:
TOP 5 ELEMENTS
![image](https://github.com/user-attachments/assets/8833e3fc-6161-4af6-ac2b-5362d02a95b9)
![image](https://github.com/user-attachments/assets/ca67c56b-7153-4500-a23c-251b28c007c8)

NULL DATA 
![image](https://github.com/user-attachments/assets/110ac27c-0664-4162-b53e-8e9ee26a8802)

DATA DUPLICATE
![image](https://github.com/user-attachments/assets/0b8600ce-7d43-424f-ab5e-f8b221cb6537)

DATA STATUS
![image](https://github.com/user-attachments/assets/1b406b54-cb25-4022-a999-705c633febd1)
![image](https://github.com/user-attachments/assets/5e4820d1-7ab9-4bcc-9bd1-95a77b7fcd32)
![image](https://github.com/user-attachments/assets/0618847f-d761-43c5-9aef-ec38938d7463)

Y_PREDICTION ARRAY
![image](https://github.com/user-attachments/assets/afb85c66-1bc8-408d-a2b6-77a0c9527e2e)

ACCURACY VALUE
![image](https://github.com/user-attachments/assets/00c4f6b4-e096-4109-bb32-a196b32ee19b)

CONFUSION ARRAY
![image](https://github.com/user-attachments/assets/4f3b18b2-0f2f-4f0a-a92b-6d585786fcf0)

CLASSIFICATION REPORT
![image](https://github.com/user-attachments/assets/2b7c2240-c4e5-469a-81c6-4a3a35796e3f)

PREDICTION ARRAY
![image](https://github.com/user-attachments/assets/4efa437f-a765-4545-93d1-211477386542)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
