import numpy as np

def read_data(path):
    data = np.loadtxt(path, delimiter=' ')
    X = data[:, :-10]
    y = fix_labels(data[:,-10:])
    return X,y

def fix_labels(labels):
    return [label.index(1) for label in labels]
    

#main
X, y = read_data(data.txt)
print(X, y, sep='\n')