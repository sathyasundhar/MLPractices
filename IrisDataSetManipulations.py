from sklearn.datasets import load_iris

#Loading input data from sklearn.datasets package

iris=load_iris()

from sklearn.model_selection import train_test_split

x = iris.data
y = iris.target

#Spliting training and test data from the given data set on ratio 8:2 respectively
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4 )

from sklearn.neighbors import KNeighborsClassifier

# for K value = 1
knnClassifer = KNeighborsClassifier(n_neighbors=1)
knnClassifer.fit(x_train, y_train)
knnPrediction = knnClassifer.predict(x_test)

#accuracy score calculation and classification report generation for K=1
from sklearn.metrics import classification_report,accuracy_score
print('Classification report')
print(classification_report(y_test, knnPrediction))
print('Accuracy when K=1 is' , accuracy_score(knnPrediction,y_test)*100)

# for K value = 5
knnClassifer = KNeighborsClassifier(n_neighbors=5)
knnClassifer.fit(x_train, y_train)
knnPrediction = knnClassifer.predict(x_test)

#accuracy score calculation and classification report generation for K=5
print('\n\nClassification report')
print(classification_report(y_test, knnPrediction))
print('Accuracy when K=5 is' , accuracy_score(knnPrediction,y_test)*100)

# for K value = 10
knnClassifer = KNeighborsClassifier(n_neighbors=10)
knnClassifer.fit(x_train, y_train)
knnPrediction = knnClassifer.predict(x_test)

#accuracy score calculation and classification report generation for K=10
print('\n\nClassification report')
print(classification_report(y_test, knnPrediction))
print('Accuracy when K=10 is' , accuracy_score(knnPrediction,y_test)*100)
