from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pandas as pd

df= pd.read_excel('ANZ synthesised transaction dataset.xlsx')
x=df[['age','gender','balance']]
y=df['amount']
hotEncoding=pd.get_dummies(x['gender'])

x=pd.concat([x,hotEncoding],axis=1)
x.drop(['gender','M'], axis=1,inplace=True)   #dropping M for rank matrix issue for inversing
# Make and fit the linear regression model
# TODO: Fit the model and assign it to the model variable
model = LinearRegression()
model.fit(x,y)
# Make a prediction using the model

Intercept=model.intercept_
Coefficients=model.coef_

print("Intercept ",Intercept,"Coeff ",Coefficients)
samplePrediction= [(26,25.39,1)] #1 for gender F
# TODO: Predict housing price for the sample_house
prediction = model.predict(samplePrediction)
print("Sample Input: ",samplePrediction)
print("Prediction: ",prediction)