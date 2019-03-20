import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.tree import DecisionTreeClassifier as dt
from sklearn.linear_model import Perceptron as pct
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.model_selection import LeaveOneOut as loo, cross_validate as cv
from scipy.stats import ttest_ind

def read_data(path):
    data = np.loadtxt(path, delimiter=',')
    features = data[:,1:]
    labels = data[:,0]
    return features, labels

def show_stats(lx, ly, classifier, validation, name):
    print(name)
    scores = []
    for i in range(len(lx)):
        stats = cv(classifier, lx[i], ly[i], cv=validation)
        scores.append(sum(stats['test_score'])/len(stats['test_score']))
    for score in scores:
        print('monk', i, ':', score)    
    print('-----------------------------')
    return sum(scores) / len(scores)


x1, y1 = read_data("monks-1.csv")
x2, y2 = read_data("monks-2.csv")
x3, y3 = read_data("monks-3.csv")

feats = [x1, x2, x3]
labs = [y1, y2, y3]

print('***Using 3-fold validation***')

worst = show_stats(feats, labs, pct(max_iter=100, tol=0), 3, 'perceptron')
best = show_stats(feats, labs, dt(max_depth=10), 3, 'decision tree')
show_stats(feats, labs, knn(n_neighbors=3), 3, 'K-nearest-neighbors')
show_stats(feats, labs, gnb(), 3, 'Gaussian Naive Bayes')

print('t test between perceptron and decision tree:', ttest_ind(worst, best))

print('***Using Leave-one-out***')

worst = show_stats(feats, labs, pct(max_iter=50, tol=0), loo(), 'perceptron')
best = show_stats(feats, labs, dt(max_depth=10), loo(), 'decision tree')
show_stats(feats, labs, knn(n_neighbors=3), loo(), 'K-nearest-neighbors')
show_stats(feats, labs, gnb(), loo(), 'Gaussian Naive Bayes')

print('t test between perceptron and decision tree:', ttest_ind(worst, best))