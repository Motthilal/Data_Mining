# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:23:34 2019

@author: motth
"""

import sys
import os
import pickle as pk
import pandas as pd
import numpy as np
from grp_22_train import feat_extraction 
from sklearn.metrics import accuracy_score


def test_script(file_name):
    
    #LOAD THE PICKLE FILES IN THE DIR
    with open("scaling_mdl.pkl", 'rb') as f:
        pca_feat_scale = pk.load(f)
    with open("pca_mdl.pkl", 'rb') as f:
        pca = pk.load(f)  
        
    with open("svm_mdl.pkl", 'rb') as f:
        support_vector_machine = pk.load(f)
        
    with open("rd_mdl.pkl", 'rb') as f:
        random_forest = pk.load(f)

    with open("knn_mdl.pkl", 'rb') as f:
        KNN = pk.load(f)

    with open("dtree_mdl.pkl", 'rb') as f:
        decision_tree = pk.load(f)


      
   
    test_data = pd.read_csv(file_name, header=None)
    

    actual_features = feat_extraction(test_data)
    
    
    test = actual_features.iloc[:].values
    
    train = pca_feat_scale.transform(test) 
    
    
    transformed_data = pca.transform(train)  
    
    svm_pred = support_vector_machine.predict(transformed_data)
    np.savetxt("SVM_pred.csv", svm_pred, '%d', ",")

    knn_pred = KNN.predict(transformed_data)
    np.savetxt("KNN_pred.csv", knn_pred, '%d', ",")
    
    dtree_pred = decision_tree.predict(transformed_data)
    np.savetxt("DecisionTree_pred.csv", dtree_pred, '%d', ",")
    
    rforest_pred = random_forest.predict(transformed_data)
    np.savetxt("RandomForest_pred.csv", rforest_pred, '%d', ",")
    
    

    
if __name__ == "__main__":
    
    f_name = sys.argv[1]
    test_script(f_name)
