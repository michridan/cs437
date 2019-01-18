# HW code for CptS 437, adapted from assignment examples
import math
import collections
import numpy

# Read a data file in csv format, separate into features and class arrays
def read_data(type):
   if type == 'train':
      data = numpy.loadtxt(fname='/content/gdrive/My Drive/ML/lectures/traindata.csv', delimiter=',')
   else:
      data = numpy.loadtxt(fname='/content/gdrive/My Drive/ML/lectures/testdata.csv', delimiter=',')
   X = data[:,:-1] # features are all values but the last on the line
   y = data[:,-1] # class is the last value on the line
   return X, y


def entropy(pos, neg):
   if pos + neg == 0:
      return 0
   pf = pos / float(pos + neg)
   nf = neg / float(pos + neg)
   if pf == 0:
     term1 = 0
   else:
     term1 = -pf * math.log(pf, 2.0)
   if nf == 0:
     term2 = 0
   else:
     term2 = -nf * math.log(nf, 2.0)
   entropy = term1 + term2
   return entropy
 
def find_gain(groups):
   splits = [[0,0], [0,0]]
   pos = 0
   neg = 0
   i = 0
   for group in groups:
      for attr in group:
         if attr[-1] == 0:
            splits[i][0] += 1
            neg += 1
         else:
            splits[i][0] += 1
            pos += 1
      i += 1

   start = entropy(pos, neg)
   sum = start
   for feature in splits:
      size = float(feature[0] + feature[1]) / float(pos + neg)
      feature_entropy = entropy(feature[0], feature[1])
      sum -= size * feature_entropy
   return sum


# Create child splits for a node or make a leaf node
def split(node, max_depth, depth):
   left, right = node['groups']
   del(node['groups'])
   # check for a no split
   if not left or not right:
      node['left'] = node['right'] = create_leaf(left + right)
      return
   # check for max depth
   if depth >= max_depth:
      node['left'], node['right'] = create_leaf(left), create_leaf(right)
      return
   node['left'] = select_attribute(left)
   split(node['left'], max_depth, depth+1)
   node['right'] = select_attribute(right)
   split(node['right'], max_depth, depth+1)


# split the dataset based on an attribute and attribute value
def test_split(index, value, dataset):
   left, right = list(), list()
   for row in dataset:
      if row[index] < value:
         left.append(row)
      else:
         right.append(row)
   return left, right


# Select the best split point for a dataset
def select_attribute(dataset):
   class_values = list(set(row[-1] for row in dataset))
   b_index, b_value, b_score, b_groups = 999, 999, -999, None
   for index in range(len(dataset[0])-1):
      for row in dataset:
         groups = test_split(index, row[index], dataset)
         gain = find_gain(groups)
         if gain > b_score:
            b_index, b_value, b_score, b_groups = index, row[index], gain, groups
   return {'index':b_index, 'value':b_value, 'groups':b_groups}


# Create a leaf node class value
def create_leaf(group):
   outcomes = [row[-1] for row in group]
   return max(set(outcomes), key=outcomes.count)


# Build a decision tree
def build_tree(train, max_depth):
   root = select_attribute(train)
   split(root, max_depth, 1)
   return root
  
  
# Print a decision tree
def print_tree(node, depth=0):
   if depth == 0:
      print 'Tree:'
   if isinstance(node, dict):
      print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
      print_tree(node['left'], depth+1)
      print_tree(node['right'], depth+1)
   else:
      print('%s[%s]' % ((depth*' ', node)))
      

# Make a prediction with a decision tree
def predict(node, row):
   if row[node['index']] < node['value']:
      if isinstance(node['left'], dict):
         return predict(node['left'], row)
      else:
         return node['left']
   else:
      if isinstance(node['right'], dict):
         return predict(node['right'], row)
      else:
         return node['right']

        
if __name__ == "__main__":
   dataset = numpy.loadtxt(fname='har.csv', delimiter=',')
   dataset = dataset.tolist()
   train = dataset[:int(2 * len(dataset) / 3)]
   test = dataset[int(2 * len(dataset) / 3):]
   tree = build_tree(train, 3)
   print_tree(tree)
   true = 0
   total = 0
   for row in test:
      prediction = predict(tree, row)
      if row[-1]:
         true += 1
      total += 1
   print('%d true positive/negatives out of %d' % (true, total))      