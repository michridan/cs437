import collections
import numpy
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklean import metrics
from sklearn.model_selection import train_test_split

classifiers = [
    (DummyClassifier(strategy='most_frequent'), 'Simple Majority'),
    (DecisionTreeClassifier(criterion='entropy'), "Decision Tree"),

]

def read_data():



def sklearn_majority_train(X, y):
    classifier = DummyClassifier(strategy='most_frequent')
    classifier.fit(X, y)
    return classifier

def sklearn_majority_test(classifier, X, y):
    newlabels = classifier.predict(X)
    print 'accuracy for simple maj classifier:' metrics.accuracy_score(y, newlabels)
    print metrics.confusion_matrix(y, newlabels)


if __name__ == "__main__":
    X, y = read_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    for clf, name in classifiers:
        clf.fit(X_train, y_train)
        newlabels = clf.predict(X_test)
        print 'classifier:', name 
        print 'accuracy:', metrics.accuracy_score(y, newlabels)
        print metrics.confusion_matrix(y, newlabels)