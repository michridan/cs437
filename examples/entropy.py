import math

def entropy(pos, neg):
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

def gain(pos, neg, splits):
	start = entropy(pos, neg)
	sum = start
	for feature in splits:
		size = float(feature[0] + feature[1]) / float(pos + neg)
		feature_entropy = entropy(feature[0], feature[1])
		sum -= size * feature_entropy
	return sum

print 'entropy', entropy(6, 0)
print 'entropy', entropy(9, 5)

print 'outlook', gain(9, 5, ([2, 3], [4, 0], [3, 2]))
print '  -> temp', gain(2, 3, ([0, 2], [1, 1], [1, 0]))
