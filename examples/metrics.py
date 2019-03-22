from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib as plt

breast_cancer = load_breast_cancer
x = breast_cancer.data
y = breast_cancer.target

x_train, x_test, y_train, y_test = train_test_split() #stuff goes here

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print('accuracy lr', metrics.accuracy_score(y_test, y_pred))

y_pred_proba = clf.predict_proba(x_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=)
