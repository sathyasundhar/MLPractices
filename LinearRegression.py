#Assignment 4- question 2

import pandas as pd

#Please change the file location while running the program

data = pd.read_csv("G:\Sathya\Sem 3\ML\Assignment\HW04\petrol_consumption.csv")


data.head()


# x and y variable definition

X = data[['Petrol_tax', 'Average_income', 'Paved_Highways',
       'Population_Driver_licence(%)']]
y = data['Petrol_Consumption']

#import statements for data split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#import statemet for LinearRegression package

from sklearn.linear_model import LinearRegression
r = LinearRegression()
r.fit(X_train, y_train)

#Coefficient value

coefficient_dataframe = pd.DataFrame(r.coef_, X.columns, columns=['Coefficient'])
print(coefficient_dataframe)

# y prediction value

y_prediction = r.predict(X_test)

# Data frame variable intialisation 

op = pd.DataFrame({'Actual': y_test, 'Predicted': y_prediction})
print(op)





