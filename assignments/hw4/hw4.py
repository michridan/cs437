import numpy as np
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.model_selection import cross_validate as cv
from scipy.stats import ttest_ind
import copy

def split_data(data):
    features = data[:,:-1]
    labels = data[:,-1]
    return features, labels

def discretize_frequency(data):
    cpy = copy.deepcopy(data)
    step = len(cpy) / 10
    for i in range(len(cpy[0]) - 1):
        if not isinstance(cpy[0,i], (int, float)):
            continue
        sorted(cpy, key=lambda n: n[i])
        j = 0
        for x in range(10):
            while j < (x + 1) * step:
                cpy[j, i] = x
                j+=1
    print(cpy)
    return cpy

def discretize_interval(data):
    cpy = copy.deepcopy(data)
    for i in range(len(cpy[0]) - 1):
        if not isinstance(cpy[0,i], (int, float)):
            continue
        step = max(cpy[:,i]) / 10
        for value in cpy:
            for x in range(10):
                if value[i] >= x * step and value[i] < (x + 1) * step:
                    value[i] = x
                    break
            else:
                value[i] = 9
    print(cpy)
    return cpy

def show_stats(x, y, classifier, validation, name):
    print(name)
    stats = cv(classifier, x, y, cv=validation)
    score = (sum(stats['test_score'])/len(stats['test_score']))
    print('Accuracy :', score)    
    print('-----------------------------')
    return stats

def one_vs_all_train(x, y, classifier, possible_labels):
    f = []
    for label in possible_labels:
        d = [1 if i == label else 0 for i in y]
        f.append(classifier.fit(x, d))
    return f

def one_vs_all_test(x, y, f):
    predictions = []
    for classifier in f:
        scores = classifier.predict_proba(x)[:,0]
        print(scores)
        predictions.append(scores)
    predictions = np.array(predictions).T
    predictions = [guess.tolist().index(max(guess)) for guess in predictions]
    pos = [int(predict == trueval) for predict, trueval in zip(predictions, y)]
    acc = sum(pos) / len(pos)
    print(acc)
    return

data = (np.loadtxt('smarthome.csv', delimiter=','))
x, y = split_data(data)
reg_stats = show_stats(x, y, gnb(), 10, 'Basic Gaussian Naive Bayes')

disc = discretize_interval(data)
x, y = split_data(disc)
disc_stats = show_stats(x, y, gnb(), 10, "Discretized Naive Bayes")

tval, pval = ttest_ind(reg_stats['test_score'], disc_stats['test_score'])
print("tval", tval, "pval", pval)

x_train = x[0 : int(len(x) * 2 / 3)]
x_test = x[int(len(x) * 2 / 3) : -1]
y_train = y[0 : int(len(y) * 2 / 3)]
y_test = y[int(len(x) * 2 / 3) : -1]

e = one_vs_all_train(x_train, y_train, gnb(), range(8))
one_vs_all_test(x_test, y_test, e)