#######################################################################
###########		GROUP 22		#######################################
#######################################################################

Files submitted for Assignment 2:
 
grp_22_train.py		|	Training script 
grp_22_test.py		| 	Testing script
cross_validtn.py	|	K-fold training and testing - Accuracy, F1_score, Recall and Precision
Eigen_Vectors.csv 	| 	PCA - Eigen vectors
pca_mdl.pkl			|	Pickle PCA model file
scaling_mdl.pkl 	|	Pickle Scaling model file
dtree_mdl.pkl 		|	Pickle Decision tree model file
knn_mdl.pkl 		|	Pickle KNN model file
svm_mdl.pkl 		|	Pickle Support Vector Machine model file
rd_mdl.pkl 			|	Pickle Random Forest model file
Real_output.csv 	|	K-fold output file

########################################################################

Submission:
Aravind Thillai Villalan| 1215121258 | athillai@asu.edu	|SVM
Motthilal Baskaran 		| 1215168292 | mbaskar2@asu.edu	|Random Forest
Vaidhehi Vasudevan 		| 1215127381 | vvasude7@asu.edu	|KNN
Sandhya Chandrasekaran 	| 1215159426 | schand61@asu.edu	|Decision Trees
########################################################################

Instructions for testing the test script in Windows:

All the contents of the zip file should be stored inside "....\DM_grp_22")

--> Extract the zip folder 'DM_grp_22' in the current working directory of the system.  

--> Open Anaconda prompt in windows and change to the folder directory:

cd DM_grp_22

--> To see the accuracy, f1-score, recall and precision - Run the cross_validtn.py using the following command:

python cross_validtn.py

--> To run the Testing.py file, use the following command:

python grp_22_test.py <test_file.csv>

CSV Files with labels will be generated in the current folder with the respective classifier names.

If running the files in MacOS then the directory path of the file should be changed in cross_validtn, grp_22_train and grp_22_test to the respective file locations.
