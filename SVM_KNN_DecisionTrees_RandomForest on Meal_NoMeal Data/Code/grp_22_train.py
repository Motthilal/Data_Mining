
import numpy as np
import pandas as pd
import os
import tsfresh.feature_extraction.feature_calculators as ts
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pickle
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def data_pre_processing():

    dt1 = pd.read_csv(r"data\mealData1.csv", usecols = [i for i in range(0,30)], header=None)
    dt2 = pd.read_csv(r"data\mealData2.csv", usecols = [i for i in range(0,30)], header=None)
    dt3 = pd.read_csv(r"data\mealData3.csv", usecols = [i for i in range(0,30)], header=None)
    dt4 = pd.read_csv(r"data\mealData4.csv", usecols = [i for i in range(0,30)], header=None)
    dt5 = pd.read_csv(r"data\mealData5.csv", usecols = [i for i in range(0,30)], header=None)
    
    dt6 = pd.read_csv(r"data\Nomeal1.csv", usecols = [i for i in range(0,30)], header=None)
    dt7 = pd.read_csv(r"data\Nomeal2.csv", usecols = [i for i in range(0,30)], header=None)
    dt8 = pd.read_csv(r"data\Nomeal3.csv", usecols = [i for i in range(0,30)], header=None)
    dt9 = pd.read_csv(r"data\Nomeal4.csv", usecols = [i for i in range(0,30)], header=None)
    dt10 = pd.read_csv(r"data\Nomeal5.csv", usecols = [i for i in range(0,30)], header=None)

    meal_dataset = dt1.append([dt2, dt3,dt4, dt5])
    nomeal_dataset = dt6.append([dt7, dt8,dt9, dt10])

    meal_dataset.reset_index(drop=True, inplace=True)
    nomeal_dataset.reset_index(drop=True, inplace=True)

    meal_dataset = meal_dataset.dropna(how='all')
    nomeal_dataset = nomeal_dataset.dropna(how='all')

    meal_dataset['Class'] = 1
    nomeal_dataset['Class'] = 0

    full_dataset = meal_dataset.append([nomeal_dataset])
    full_dataset.columns = ['Time1',   'Time2',    'Time3',    'Time4',    'Time5',    'Time6',    'Time7',    'Time8',    'Time9',    'Time10',   'Time11',   'Time12',   'Time13',   'Time14',   'Time15',   'Time16',   'Time17',   'Time18',   'Time19',   'Time20',   'Time21',   'Time22',   'Time23',   'Time24',   'Time25',   'Time26',   'Time27',   'Time28',   'Time29',   'Time30', 'Class']

    full_dataset.reset_index(drop=True, inplace= True)
    meal_dataset = meal_dataset.drop(['Class'], axis=1)
    nomeal_dataset = nomeal_dataset.drop(['Class'], axis=1)


    X = meal_dataset.iloc[:,:].values
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=1)
    imputer = imputer.fit(X[:,:])
    X[:,:] = imputer.transform(X[:,:])

    meal_dataset = pd.DataFrame(X)


    Y = nomeal_dataset.iloc[:,:].values
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=1)
    imputer = imputer.fit(Y[:,:])
    Y[:,:] = imputer.transform(Y[:,:])

    nomeal_dataset = pd.DataFrame(Y)
    
    meal_dataset['Class'] = 1
    nomeal_dataset['Class'] = 0
    
    real_output = meal_dataset.append([nomeal_dataset])
    real_output.to_csv("Real_output.csv", index = False)
    
    meal_dataset.drop(columns = 'Class', axis = 1, inplace = True)
    nomeal_dataset.drop(columns = 'Class', axis = 1, inplace = True)
    
    meal_feature_data = feat_extraction(meal_dataset)
    nomeal_feature_data = feat_extraction(nomeal_dataset)
    
    
    meal_feature_data.iloc[:,:] = np.round(meal_feature_data.iloc[:,:],2)
    nomeal_feature_data.iloc[:,:] = np.round(nomeal_feature_data.iloc[:,:],2)
    
    
    meal_feature_data['Class'] = 1
    nomeal_feature_data['Class'] = 0
    
    feat_data = meal_feature_data.append([nomeal_feature_data])
    feat_data.reset_index(drop=True, inplace= True)
    
    
    return feat_data

def feat_extraction(dataset):
    
    feat_dataset = pd.DataFrame(index = np.arange(len(dataset)))
    
    #Calculated columns
    feat_dataset['CGM_Min'] = dataset.min(axis=1)
    feat_dataset['CGM_Max'] = dataset.max(axis=1)
    
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

    return feat_dataset
    

def Standard_Scaler(X_train):
    
    pca_feat = StandardScaler()
    X_train = pca_feat.fit_transform(X_train) 
    
    pickle_fname = "scaling_mdl.pkl"
    
    with open(pickle_fname,'wb') as f:
        pickle.dump(pca_feat, f)

    return X_train

def Pca_imp(X_train):
    
    pca = PCA(0.95)
    pca_counted_weights = pca.fit(X_train)
    pca_res = pca.fit_transform(X_train)
    
    new_file = pd.DataFrame((pca.components_).T)

    new_file.to_csv('Eigen_Vectors.csv',index = False, header = None)
    
    pickle_fname = "pca_mdl.pkl"
    with open(pickle_fname, 'wb') as f:
        pickle.dump(pca_counted_weights, f)
    
    return pca_res



def svm_support_vector_machine(X_train, Y_train):
    
    clf_svm=svm.SVC(kernel = 'linear', C=1)
    clf_svm.fit(X_train,Y_train)
    
    pickle_fname = "svm_mdl.pkl"
    with open(pickle_fname, 'wb') as f:
        pickle.dump(clf_svm, f)



def dt_decision_trees(X_train, Y_train):
    
    clf_dtree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.1)
    clf_dtree.fit(X_train,Y_train)

    pickle_fname = "dtree_mdl.pkl"
    with open(pickle_fname, 'wb') as f:
        pickle.dump(clf_dtree, f)

def rf_random_forest(X_train, Y_train):
    
    clf_rforest=RandomForestClassifier(n_estimators=100, max_depth=4, max_features=3)
    clf_rforest.fit(X_train,Y_train)

    pickle_fname = "rd_mdl.pkl"
    with open(pickle_fname, 'wb') as f:
        pickle.dump(clf_rforest, f)



def KNN(X_train, Y_train):
    
    clf_knn = KNeighborsClassifier(n_neighbors=10, p=2)
    clf_knn.fit(X_train, Y_train)
    
    pickle_fname = "knn_mdl.pkl"
    with open(pickle_fname, 'wb') as f:
        pickle.dump(clf_knn, f)




     
def Classifier_class(features):
        
    X_ = features.iloc[:,:-1].values
    Y_ = features.iloc[:,-1].values
    
    X_train = Standard_Scaler(X_)        
    X_train = Pca_imp(X_train)
    
    svm_support_vector_machine(X_train, Y_)
    rf_random_forest(X_train, Y_)
    KNN(X_train, Y_)
    dt_decision_trees(X_train, Y_)


if __name__ == "__main__":
    
    Classifier_class(data_pre_processing())
     
