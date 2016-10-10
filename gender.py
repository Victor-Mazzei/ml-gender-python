from sklearn import tree
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB

# data train [height,weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#classifiers
clf = tree.DecisionTreeClassifier()
clf1 = svm.SVC()
clf2 = neighbors.KNeighborsClassifier()
clf3 = GaussianNB()
#train model
clf = clf.fit(X,Y)
clf1 = clf1.fit(X,Y)
clf2 = clf2.fit(X,Y)
clf3 = clf3.fit(X,Y)

_X=[[184,84,44],[198,92,48],[183,83,44],[166,47,36],[170,60,38],[172,64,39],[182,80,42],[180,80,43]]
_Y=['male','male','male','female','female','female','male','male']

#prediction
prediction = clf.predict(_X)
prediction1 = clf1.predict(_X)
prediction2 = clf2.predict(_X)
prediction3 = clf3.predict(_X)



print("SVM : ",accuracy_score(_Y,prediction1))
print("DecisionTree : ",accuracy_score(_Y,prediction))
print("KNeighborsClassifier : ",accuracy_score(_Y,prediction2))
print("Naive : ",accuracy_score(_Y,prediction3))


