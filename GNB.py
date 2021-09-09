#Assignment 5-Part 2
from sklearn.datasets import load_breast_cancer

#Load breast cancer dataset
BCDataSet = load_breast_cancer()

#Data split for test and traing
from sklearn.model_selection import train_test_split
x, y = BCDataSet.data, BCDataSet.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#GaussianNB Class import and object creations

from sklearn.naive_bayes import GaussianNB
gaussianNBObj= GaussianNB()
y_prediction = gaussianNBObj.fit(x_train, y_train).predict(x_test)

#Accuracy calculation after y prediction

from sklearn.metrics import classification_report, accuracy_score
print('Accuracy of the Gaussian Naive Bayesian model is ' , accuracy_score(y_prediction,y_test)*100)

#report generation

print(classification_report(y_prediction,y_test))

#LinearDiscriminantAnalysis class import and object creation

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


LDAObj = LinearDiscriminantAnalysis()
y_prediction=LDAObj.fit(x_train, y_train).predict(x_test)

#Accuracy calculation after y prediction

print('Accuracy of the linear discriminant analysis model is ' , accuracy_score(y_prediction,y_test)*100)

#report generation

print(classification_report(y_prediction,y_test))



