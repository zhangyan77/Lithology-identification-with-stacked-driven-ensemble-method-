import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from  sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

class MidpointNormalize(Normalize):

	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))


def getData(path):
	data1=np.loadtxt(path,skiprows=1)
	data2=data1[:,1:-1]
	data2[:, -2] = np.log10(data2[:, -2])
	data,label=data2[:,:-1],data2[:,-1]
	depth = data1[:, 0]
	return depth, data, label


def sklearn_confusion_matrix(ytest, ymodel):
	from sklearn.metrics import confusion_matrix
	import seaborn as sns
	import matplotlib.pyplot as plt

	mat = confusion_matrix(ytest, ymodel)
	mat1 = my_lib.MakeConfusionMatrixWithACC(mat)
	# print(mat)
	sns.heatmap(mat1, square=True, annot=True, cbar=False)
	plt.xlabel('predicted value')
	plt.ylabel('true value')
	plt.savefig('well1.png', dpi = 1000)
	plt.clf()


if __name__ == '__main__':
	depth1, x_allData,y_allLable = getData('TrainSet.txt')
	scaler = MinMaxScaler()
	scaler.fit(x_allData)
	scaler_data = scaler.transform(x_allData)
	allData = np.column_stack((scaler_data, y_allLable))
	X_trainval, X_test, y_trainvalLable, y_testLable = train_test_split(allData[:, :-1], allData[:, -1],
																		test_size=0.2, random_state=1)
	X_train, X_valid, y_trainLable, y_validLable = train_test_split(X_trainval, y_trainvalLable, test_size=0.3,
																	random_state=1)
	print('Number of samples£º', scaler_data.shape)
	print('Number of training and validating sets£º', X_trainval.shape)
	print('Number of training sets£º', X_train.shape)
	print('Number of validating sets£º', X_valid.shape)
	print('Number of testing sets£º', X_test.shape)

	'''===============SVR=================='''
	best_score = 0
	f1 = open('TrainSet_SVM_TestSets.txt', 'w')
	print('g', end='\t', file=f1)
	print('C', end='\t', file=f1)
	for i in range(1,11):
		print('NO' + str(i), end="\t", file=f1)
	print('Accuracy of 10 folds', end='\t', file=f1)
	print('Variances', end='\n', file=f1)
	f2 = open('TrainSet_SVM_TrainSets.txt', 'w')
	print('g', end='\t', file=f2)
	print('C', end='\t', file=f2)
	print('Accuracy', end='\n', file=f2)

	C_range = np.logspace(-5, 5, 11)
	gamma_range = np.logspace(-7, 3, 11)
	scores_mean = []
	for gamma in gamma_range:
		for C in C_range:
			svm = SVC(gamma=gamma, C=C)
			svm.fit(X_train, y_trainLable)
			scores_train = svm.score(X_train, y_trainLable)
			scores = cross_val_score(svm, X_valid, y_validLable, cv=10)
			print(gamma, end='\t', file=f2)
			print(C, end='\t', file=f2)
			print(scores_train, end='\n', file=f2)
			print(gamma, end='\t', file=f1)
			print(C, end='\t', file=f1)
			for j in range(0, 10):
				print(scores[j], end='\t', file=f1)
			print(scores.mean(), end='\t', file=f1)
			print(scores.std(), end='\n', file=f1)
			scores_mean.append(scores.mean())
			score = np.mean(scores)
			if score > best_score:
				best_score = score
				best_parameters = {'C': C, 'gamma': gamma}

	f2.close()
	f1.close()
	print('cross_val_score:',best_parameters, best_score)
	svmf = SVC(**best_parameters)
	svmf.fit(X_test, y_testLable)
	print('cross_val_score on testing sets:',svmf.score(X_test,y_testLable))


	# '''===============BP=================='''
	#
	best_score = 0
	f1 = open('TrainSet_BP_TestSets.txt', 'w')
	print('Num_Hidden', end='\t', file=f1)
	for i in range(1, 11):
		print('NO' + str(i), end="\t", file=f1)
	print('Accuracy of 10 folds', end='\t', file=f1)
	print('Variances', end='\n', file=f1)
	f2 = open('TrainSet_BP_TrainSets.txt', 'w')
	print('Num_Hidden', end='\t', file=f2)
	print('Accuracy', end='\n', file=f2)

	hidden_layer = range(100,400,4)
	for layers in hidden_layer:
		BP = MLPClassifier(activation='logistic', hidden_layer_sizes=layers)
		BP.fit(X_train, y_trainLable)
		scores_train = BP.score(X_train, y_trainLable)
		print(scores_train)
		scores = cross_val_score(BP, X_valid, y_validLable, cv=10)
		print(layers, end='\t', file=f2)
		print(scores_train, end='\n', file=f2)
		print(layers, end='\t', file=f1)
		for j in range(0, 10):
			print(scores[j], end='\t', file=f1)
		print(scores.mean(), end='\t', file=f1)
		print(scores.std(), end='\n', file=f1)
		score = np.mean(scores)
		if score > best_score:
			best_score = score
			best_parameters = {'hidden_layer_sizes': layers}

	f1.close()
	f2.close()
	print('cross_val_score:',best_parameters, best_score)
	bpf = MLPClassifier(activation='logistic', hidden_layer_sizes=best_parameters.values())
	bpf.fit(X_test, y_testLable)
	print('cross_val_score on testing sets:',bpf.score(X_test,y_testLable))
	#
	# '''===============DT=================='''
	best_score = 0
	f1 = open('TrainSet_DT_TestSets.txt', 'w')
	print('max_depth', end='\t', file=f1)
	print('min_samples_leaf', end='\t', file=f1)
	for i in range(1,11):
		print('NO' + str(i), end="\t", file=f1)
	print('Accuracy of 10 folds', end='\t', file=f1)
	print('Variances', end='\n', file=f1)
	f2 = open('TrainSet_DT_TrainSets.txt', 'w')
	print('max_depth', end='\t', file=f2)
	print('min_samples_leaf', end='\t', file=f2)
	print('Accuracy', end='\n', file=f2)

	max_depth_range = range(5,102,2)
	min_samples_leaf_range = range(5,102,2)
	for max_depth in max_depth_range:
		for min_samples_leaf in min_samples_leaf_range:
			dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf = min_samples_leaf)
			dt.fit(X_train, y_trainLable)
			scores_train = dt.score(X_train, y_trainLable)
			scores = cross_val_score(dt, X_valid, y_validLable, cv=10)
			print(max_depth, end='\t', file=f2)
			print(min_samples_leaf, end='\t', file=f2)
			print(scores_train, end='\n', file=f2)
			print(max_depth, end='\t', file=f1)
			print(min_samples_leaf, end='\t', file=f1)
			for j in range(0, 10):
				print(scores[j], end='\t', file=f1)
			print(scores.mean(), end='\t', file=f1)
			print(scores.std(), end='\n', file=f1)
			score = np.mean(scores)
			if score > best_score:
				best_score = score
				best_parameters = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}

	f2.close()
	f1.close()
	print('cross_val_score:',best_parameters, best_score)
	dtf = DecisionTreeClassifier(**best_parameters)
	dtf.fit(X_test, y_testLable)
	print('cross_val_score on testing sets:',dtf.score(X_test,y_testLable))