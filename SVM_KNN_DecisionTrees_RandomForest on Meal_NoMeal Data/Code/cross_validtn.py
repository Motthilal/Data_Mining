import numpy as np
import pandas as pd
import tsfresh.feature_extraction.feature_calculators as ts
from sklearn.model_selection import KFold

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score




Real_output = pd.read_csv(r"Real_output.csv")
dataset = Real_output.drop(columns = 'Class', axis = 1)

##Create dataframe for feature selection
feat_dataset=pd.DataFrame()
#Calculated columns
feat_dataset['CGM_Min'] = dataset.min(axis=1)
feat_dataset['CGM_Max'] = dataset.max(axis=1)
#Reset index after merging different files into one
feat_dataset.reset_index(drop=True, inplace=True)
##Populate feature characteristics
##ENTROPY
feat_dataset['CGM_Entropy'] = np.nan
for i in range(len(dataset)):
   feat_dataset['CGM_Entropy'][i] = ts.sample_entropy(np.array(dataset.iloc[i,:]))
##RMS
feat_dataset['CGM_RMS'] = np.nan
for i in range(len(dataset)):
   feat_dataset['CGM_RMS'][i] = np.sqrt(np.mean(dataset.iloc[i,:]**2))
#Correlation
feat_dataset['CGM_Correlation'] = np.nan
for i in range(len(dataset)):
   feat_dataset['CGM_Correlation'][i] = ts.autocorrelation(np.array(dataset.iloc[i,:]), 1)
##Number_of_Peaks
feat_dataset['CGM_Peaks'] = np.nan
for i in range(len(dataset)):
   feat_dataset['CGM_Peaks'][i] = ts.number_peaks(np.array(dataset.iloc[i,:]),2)
#CGM Velocity
feat_dataset['CGM_Velocity'] = np.nan
for i in range(len(dataset)):
   c_list = dataset.loc[i,:].tolist()
   sum_=[]
   for j in range(1,len(c_list)):
       sum_.append(abs(c_list[j]-c_list[j-1]))
   feat_dataset['CGM_Velocity'][i] = np.round(np.mean(sum_),2)
#MinMax
feat_dataset['CGM_MinMax'] = np.nan
feat_dataset['CGM_MinMax'] = feat_dataset['CGM_Max'] - feat_dataset['CGM_Min']
##SKewness
feat_dataset['CGM_Skewness'] = np.nan
for i in range(len(dataset)):
   feat_dataset['CGM_Skewness'][i] = ts.skewness(dataset.loc[i,:])
#CGM_Displacement
feat_dataset['CGM_Displacement'] = np.nan
for i in range(len(dataset)):
   c_list = dataset.loc[i,:].tolist()
   sum_=[]
   for j in range(1,len(c_list)):
       sum_.append(abs(c_list[j]-c_list[j-1]))
   feat_dataset['CGM_Displacement'][i] = np.round(np.sum(sum_),2)
#CGM_Kurtosis
feat_dataset['CGM_Kurtosis'] = np.nan
for i in range(len(dataset)):
   feat_dataset['CGM_Kurtosis'][i] = ts.kurtosis(np.array(dataset.iloc[i,:]))
#Recurr
feat_dataset['CGM_Recur'] = np.nan
for i in range(len(dataset)):
   feat_dataset['CGM_Recur'][i] = ts.ratio_value_number_to_time_series_length(np.array(dataset.iloc[i,:]))
#Remove calculated columns
del feat_dataset['CGM_Max']
del feat_dataset['CGM_Min']


feat_dataset = feat_dataset[['CGM_Entropy', 'CGM_RMS', 'CGM_Correlation', 'CGM_Peaks','CGM_Velocity', 'CGM_MinMax', 'CGM_Skewness', 'CGM_Displacement','CGM_Kurtosis', 'CGM_Recur']]



from sklearn.preprocessing import StandardScaler

feat_dataset['Class'] = Real_output['Class']


X_=feat_dataset.iloc[:,:-1].values
Y_=feat_dataset.iloc[:,-1].values

test_svm_labels = []
test_dtree_labels = []
test_rforest_labels = []
test_knn_labels = []


pred_svm_labels = []
pred_dtree_labels = []
pred_rforest_labels = []
pred_knn_labels = []


accuracy_svm = []
accuracy_dtree = []
accuracy_rforest = []
accuracy_knn = []

f1_knn=[]
f1_svm = []
f1_dtree = []
f1_rforest = []

precision_knn=[]
precision_svm = []
precision_dtree = []
precision_rforest = []

recall_knn=[]
recall_svm = []
recall_dtree = []
recall_rforest = []


# K-Fold
from sklearn.utils.extmath import randomized_svd

#skf = StratifiedKFold(n_splits=2, random_state=None)
skf = KFold(n_splits=5, random_state=42, shuffle=True)
#skf = ShuffleSplit(n_splits=10, test_size=.30, random_state=42)

for train_index,test_index in skf.split(X_, Y_):
    
    X_train, X_test = X_[train_index], X_[test_index]
    Y_train, Y_test = Y_[train_index], Y_[test_index]
    
    pca_feat = StandardScaler()
    X_train = pca_feat.fit_transform(X_train) 
    X_test = pca_feat.transform(X_test) 
    

    pca = PCA(n_components = 5)
    principalComponents_train = pca.fit(X_train)
    
    component_matrix_train = (pca.components_).T
    dot = np.dot(X_train, component_matrix_train)
    dot2 = np.dot(X_test, component_matrix_train)
    

    # KNN
    clf_knn = KNeighborsClassifier(n_neighbors=10, p=2)
    clf_knn.fit(dot, Y_train)
    Y_pred_knn = clf_knn.predict(dot2)
    test_knn_labels.extend(Y_test)
    pred_knn_labels.extend(Y_pred_knn)
    accuracy_knn.append(accuracy_score(test_knn_labels,pred_knn_labels))
    f1_knn.append(f1_score(test_knn_labels,pred_knn_labels))
    precision_knn.append(precision_score(test_knn_labels,pred_knn_labels))
    recall_knn.append(recall_score(test_dtree_labels,pred_dtree_labels))

        
    # SVM        
    clf_svm=svm.SVC(kernel = 'rbf', gamma=0.009, C=1)
    clf_svm.fit(dot,Y_train)
    Y_pred_svm=clf_svm.predict(dot2)
    test_svm_labels.extend(Y_test)
    pred_svm_labels.extend(Y_pred_svm)
    accuracy_svm.append(accuracy_score(test_svm_labels,pred_svm_labels))
    f1_svm.append(f1_score(test_svm_labels,pred_svm_labels))
    precision_svm.append(precision_score(test_svm_labels,pred_svm_labels))
    recall_svm.append(recall_score(test_dtree_labels,pred_dtree_labels))
       

    # DecisionTree
    clf_dtree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1)
    clf_dtree.fit(dot,Y_train)
    Y_pred_dtree = clf_dtree.predict(dot2)
    test_dtree_labels.extend(Y_test)
    pred_dtree_labels.extend(Y_pred_dtree)
    accuracy_dtree.append(accuracy_score(test_dtree_labels,pred_dtree_labels))
    f1_dtree.append(f1_score(test_dtree_labels,pred_dtree_labels))
    precision_dtree.append(precision_score(test_dtree_labels,pred_dtree_labels))
    recall_dtree.append(recall_score(test_dtree_labels,pred_dtree_labels))
    
    
    # RandomForest
    clf_rforest=RandomForestClassifier(n_estimators=100, max_depth=4, max_features=3)
    clf_rforest.fit(dot,Y_train)
    Y_pred_rforest=clf_rforest.predict(dot2)
    test_rforest_labels.extend(Y_test)
    pred_rforest_labels.extend(Y_pred_rforest)
    accuracy_rforest.append(accuracy_score(test_rforest_labels,pred_rforest_labels))
    f1_rforest.append(f1_score(test_rforest_labels,pred_rforest_labels))
    precision_rforest.append(precision_score(test_rforest_labels,pred_rforest_labels))
    recall_rforest.append(recall_score(test_rforest_labels,pred_rforest_labels))
    

    
        
print("Accuracy")    
print('SVM:', np.mean(accuracy_svm))
print('KNN:', np.mean(accuracy_knn))
print('Dtree:',np.mean(accuracy_dtree))
print('RFOREST:',np.mean(accuracy_rforest))

print("\n F1_SCORE")  
print('SVM:', np.mean(f1_svm))
print('KNN:', np.mean(f1_knn))
print('Dtree:',np.mean(f1_dtree))
print('RFOREST:',np.mean(f1_rforest))

print("\n PRECISION")  
print('SVM:', np.mean(precision_svm))
print('KNN:', np.mean(precision_knn))
print('Dtree:',np.mean(precision_dtree))
print('RFOREST:',np.mean(precision_rforest))

print("\n RECALL")  
print('SVM:', np.mean(recall_svm))
print('KNN:', np.mean(recall_knn))
print('Dtree:',np.mean(recall_dtree))
print('RFOREST:',np.mean(recall_rforest))