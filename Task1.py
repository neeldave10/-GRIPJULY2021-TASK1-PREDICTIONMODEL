import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#Loading data

df=pd.read_csv('Task1.csv')
#print(df.tail())

#Checking for null values if any
#print(df.isnull().values)

#Finding correlations
col1=df['Hours']
col2=df['Scores']
correlation=col1.corr(col2)
#print(correlation)

#Plotting the two to get some more insights
sns.scatterplot(x=df['Hours'],y=df['Scores'],data=df)
#plt.show()

#Conclusion: Hours column is almost linearly increasing with scores. Let's apply the linear regression model for predictions.

hours=df.Hours.values[:,np.newaxis]
scores=df.Scores.values

#MODEL
reg=LinearRegression()
reg.fit(hours,scores)
coeff=reg.coef_
intercept=reg.intercept_
predicted_values=[reg.coef_* i + reg.intercept_ for i in hours]
plt.scatter(hours,scores)
plt.plot(hours,predicted_values)
plt.show()
print(reg.predict([[9.25]]))















