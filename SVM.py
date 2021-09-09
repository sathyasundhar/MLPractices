#Assignment 07 
#700713317 

from sklearn import datasets

datasets = datasets.load_breast_cancer()

from sklearn.model_selection import train_test_split
x, y = datasets.data, datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

from sklearn import svm

#Linear SVM classifier

Classifier = svm.SVC(kernel='linear') 

Classifier.fit(x_train, y_train)

yPrediction = Classifier.predict(x_test)

from sklearn.metrics import accuracy_score
print('Accuracy of Linear SVM model is ' , accuracy_score(yPrediction,y_test)*100)

#RBF classifier 

rbfClassifier = svm.SVC(kernel='rbf',probability=True) 

rbfClassifier.fit(x_train, y_train)

rbfYPrediction = rbfClassifier.predict(x_test)

print('Accuracy of RBF SVM model is ' , accuracy_score(rbfYPrediction,y_test)*100)