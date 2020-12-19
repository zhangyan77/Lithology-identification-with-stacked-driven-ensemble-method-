
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import My_PythonLib as my_lib

import warnings; warnings.filterwarnings(action='ignore')

def getData(path):
	data1=np.loadtxt(path,skiprows=1)
	data2=data1[:,1:-1]
	data2[:, -2] = np.log10(data2[:, -2])
	data,label=data2[:,:-1],data2[:,-1]
	depth = data1[:, 0]
	return depth, data, label

def sklearn_confusion_matrix(ytest, ymodel, name):
	mat = confusion_matrix(ytest, ymodel)
	mat1 = my_lib.MakeConfusionMatrixWithACC(mat)
	# print(mat)
	sns.heatmap(mat1, square=True, annot=True, cbar=False)
	plt.xlabel('predicted value')
	plt.ylabel('true value')
	plt.savefig('{0}.jpg'.format(name), dpi = 1000)
	plt.clf()

def StackingMethod(X, y):
    scaler = MinMaxScaler()
    scaler.fit(X)
    traffic_feature = scaler.transform(X)
    feature_train, feature_test, target_train, target_test = model_selection.train_test_split(traffic_feature, y, test_size=0.2, random_state=32)
    print('Number of samples£º', traffic_feature.shape)
    print('Number of training sets£º', feature_train.shape)
    print('Number of testing sets£º', feature_test.shape)

    clf1 = MLPClassifier(hidden_layer_sizes=296)
    clf2 = DecisionTreeClassifier(max_depth=9, min_samples_leaf=7)
    clf3 = SVC(C=1000.0, gamma=0.01)

    clf4 = RandomForestClassifier()
    clf6 = XGBClassifier()
    #clf7 = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=100)
    clf7 = GradientBoostingClassifier()

    sclf = StackingClassifier(classifiers=[clf1, clf2, clf3],
                              #use_probas=True
                              #average_probas=False
                              meta_classifier=clf4)

    clf1.fit(traffic_feature, y)
    clf2.fit(traffic_feature, y)
    clf3.fit(traffic_feature, y)
    sclf.fit(traffic_feature, y)

    clf4.fit(traffic_feature, y)
    clf6.fit(traffic_feature, y)
    clf7.fit(traffic_feature, y)

    y_testLable_bp = clf1.predict(feature_test)
    y_testLable_dt = clf2.predict(feature_test)
    y_testLable_svm = clf3.predict(feature_test)
    y_testLable_stack = sclf.predict(feature_test)
    conf_mat_bp = confusion_matrix(target_test, y_testLable_bp)
    print('BP:\n')
    print(conf_mat_bp)
    print(classification_report(target_test, y_testLable_bp))

    conf_mat_dt = confusion_matrix(target_test, y_testLable_dt)
    print('DT:\n')
    print(conf_mat_dt)
    print(classification_report(target_test, y_testLable_dt))

    conf_mat_svm = confusion_matrix(target_test, y_testLable_svm)
    print('SVM:\n')
    print(conf_mat_svm)
    print(classification_report(target_test, y_testLable_svm))

    conf_mat_stack = confusion_matrix(target_test, y_testLable_stack)
    print('STACK:\n')
    print(conf_mat_stack)
    print(classification_report(target_test, y_testLable_stack))

    sklearn_confusion_matrix(target_test, y_testLable_bp, 'Pic_testCM_BP')
    sklearn_confusion_matrix(target_test, y_testLable_dt, 'Pic_testCM_DT')
    sklearn_confusion_matrix(target_test, y_testLable_svm, 'Pic_testCM_SVM')
    sklearn_confusion_matrix(target_test, y_testLable_stack, 'Pic_testCM_STACK')

    f2 = open('STACK_CV.txt', 'w')
    for clf, label in zip([clf1, clf2, clf3, sclf], ['BP', 'DT', 'SVM', 'StackingModel']):
        scores = model_selection.cross_val_score(clf, feature_test, target_test, cv=10, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        print('======================%s====================='.format(label), end='\n', file=f2)
        for i in range(0, 10):
	        print(i, end='\t', file=f2)
	        print(scores[i], file=f2)
		print(end='\n', file=f2)




    depth, X_PRE, y_TrueLable = getData('Well1.txt')
    scaler_PRE_data = scaler.transform(X_PRE)

    y_predictedlable_bp = clf1.predict(scaler_PRE_data)
    y_predictedlable_dt = clf2.predict(scaler_PRE_data)
    y_predictedlable_svm = clf3.predict(scaler_PRE_data)
    y_Predictedlable_stack = sclf.predict(scaler_PRE_data)

    y_Predictedlable_rf = clf4.predict(scaler_PRE_data)
    y_Predictedlable_xgb = clf6.predict(scaler_PRE_data)
    y_Predictedlable_gbc = clf7.predict(scaler_PRE_data)
    print('Well1_bp:', accuracy_score(y_predictedlable_bp, y_TrueLable))
    print('Well1_dt:', accuracy_score(y_predictedlable_dt, y_TrueLable))
    print('Well1_svm:', accuracy_score(y_predictedlable_svm, y_TrueLable))
    print('Well1_stack:', accuracy_score(y_Predictedlable_stack, y_TrueLable))

    print('Well1_rf:', accuracy_score(y_Predictedlable_rf, y_TrueLable))
    print('Well1_xgb:', accuracy_score(y_Predictedlable_xgb, y_TrueLable))
    print('Well1_gbc:', accuracy_score(y_Predictedlable_gbc, y_TrueLable))

    f3 = open('CM_Suxi.txt', 'w')
    conf_mat_well_bp = confusion_matrix(y_TrueLable, y_predictedlable_bp)
    print('========================BP===================',file=f3)
    print(conf_mat_well_bp, end='\n', file=f3)
    print(classification_report(y_TrueLable, y_predictedlable_bp), end='\n', file=f3)
    # print(classification_report(y_TrueLable, y_predictedlable_bp))

    conf_mat_well_dt = confusion_matrix(y_TrueLable, y_predictedlable_dt)
    print('========================DT===================', file=f3)
    print(conf_mat_well_dt, end='\n', file=f3)
    print(classification_report(y_TrueLable, y_predictedlable_dt), end='\n', file=f3)

    conf_mat_well_svm = confusion_matrix(y_TrueLable, y_predictedlable_svm)
    print('========================SVM===================', file=f3)
    print(conf_mat_well_svm, end='\n', file=f3)
    print(classification_report(y_TrueLable, y_predictedlable_svm), end='\n', file=f3)

    conf_mat_well_stack = confusion_matrix(y_TrueLable, y_Predictedlable_stack)
    print('========================STACK===================', file=f3)
    print(conf_mat_well_stack, end='\n', file=f3)
    print(classification_report(y_TrueLable, y_Predictedlable_stack), end='\n', file=f3)

    conf_mat_well_stack = confusion_matrix(y_TrueLable, y_Predictedlable_rf)
    print('========================RF===================', file=f3)
    print(conf_mat_well_stack, end='\n', file=f3)
    print(classification_report(y_TrueLable, y_Predictedlable_rf), end='\n', file=f3)

    conf_mat_well_stack = confusion_matrix(y_TrueLable, y_Predictedlable_xgb)
    print('========================XGB===================', file=f3)
    print(conf_mat_well_stack, end='\n', file=f3)
    print(classification_report(y_TrueLable, y_Predictedlable_xgb), end='\n', file=f3)

    conf_mat_well_stack = confusion_matrix(y_TrueLable, y_Predictedlable_gbc)
    print('========================GBC===================', file=f3)
    print(conf_mat_well_stack, end='\n', file=f3)
    print(classification_report(y_TrueLable, y_Predictedlable_gbc), end='\n', file=f3)

    f1 = open('Well1_pre.txt', 'w')
    print('Depth\tAC\tCNL\tDEN\tGR\tPE\tRLLD\tLIMSTONE\tLIMSTON_BP\tLIMSTON_DT\tLIMSTON_SVM\tLIMSTON_STACK\tLIMSTON_RF\tLIMSTON_XGB\tLIMSTON_GBC', file=f1)
    for j in range(len(X_PRE)):
	    print(depth[j], end='\t', file=f1)  # Depth
	    print(X_PRE[j, 0], end='\t', file=f1)  # AC
	    print(X_PRE[j, 1], end='\t', file=f1)  # CNL
	    print(X_PRE[j, 2], end='\t', file=f1)  # DEN
	    print(X_PRE[j, 3], end='\t', file=f1)  # GR
	    print(X_PRE[j, 4], end='\t', file=f1)  # PE
	    print(np.power(10, X_PRE[j, 5]), end='\t', file=f1)  # RLLD
	    print(y_TrueLable[j], end='\t', file=f1)
	    print(y_predictedlable_bp[j], end='\t', file=f1)
	    print(y_predictedlable_dt[j], end='\t', file=f1)
	    print(y_predictedlable_svm[j], end='\t', file=f1)
	    print(y_Predictedlable_stack[j], end='\t', file=f1)
	    print(y_Predictedlable_rf[j], end='\t', file=f1)
	    print(y_Predictedlable_xgb[j], end='\t', file=f1)
	    print(y_Predictedlable_gbc[j], end='\n', file=f1)

    f1.close()

    return sclf

#========================================================
#  Ö÷³ÌÐò
#========================================================

if __name__ == '__main__':
	depth1, x_allData, y_allLable = getData('TrainSet.txt')
	model = StackingMethod(x_allData, y_allLable)