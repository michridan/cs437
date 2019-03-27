def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import VotingClassifier as vc
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA

def read_data(path):
	data = np.loadtxt(path)
	X = data[:, :-10]
	y = fix_labels(data[:,-10:])
	return X,y

def fix_labels(labels):
	return [label.tolist().index(1) for label in labels]

def show_voting_results(clfs, X, y):
	vote = vc(clfs)
	scores = cross_validate(vote, X, y, cv=3)['test_score'].tolist()
	for score in scores:
		print('fold', scores.index(score), ':', score)
	print('Average :', sum(scores)/len(scores))
    
#main
X, y = read_data("data.txt")
print("Without PCA, #components =", len(X[0]))
clf1 = mlp(alpha=0.00001, activation='relu')
clf2 = rf(n_estimators=10)
clf3 = lr(max_iter=50)
show_voting_results([('mlp', clf1), ('rf', clf2), ('lr', clf3)], X, y)

pca = PCA(n_components='mle')
X2 = pca.fit_transform(X)
print("With PCA, #components =", pca.n_components_)
show_voting_results([('mlp', clf1), ('rf', clf2), ('lr', clf3)], X2, y)