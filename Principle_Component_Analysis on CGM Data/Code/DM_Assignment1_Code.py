#Group 22

##Import required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tsfresh.feature_extraction.feature_calculators as ts

##Merge all 4 test subject datasets into one.
dataset1 = pd.read_csv("CGMSeriesLunchPat1_PP.csv")
dataset2 = pd.read_csv("CGMSeriesLunchPat2_PP.csv")
dataset3 = pd.read_csv("CGMSeriesLunchPat3_PP.csv")
dataset4 = pd.read_csv("CGMSeriesLunchPat4_PP.csv")
dataset5 = pd.read_csv("CGMSeriesLunchPat5_PP.csv")

dataset = dataset1.append([dataset2, dataset3,dataset4,dataset5])
dataset.reset_index(drop=True, inplace=True)

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
#Scale feature matrix
pca_feat = StandardScaler().fit_transform(feat_dataset)

#Implement PCA
from sklearn.decomposition import PCA 
pca = PCA()
principalComponents = pca.fit_transform(pca_feat)


ratio = pd.DataFrame(np.round(pca.explained_variance_ratio_,6),columns=['Weighted_Ratio'])
pc = pd.DataFrame(principalComponents,columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'])

#########   PLOT   FOR    Features    ###########
# # Visualising the Training set results
# feat_dataset['CGM_days'] = np.nan
# feat_dataset['CGM_days']= feat_dataset.index+1

# #Velocity
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_Velocity'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_Velocity'], color = 'blue',linewidth=4.0)
# plt.title('CGM_Velocity vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_Velocity')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #Entropy
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_Entropy'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_Entropy'], color = 'blue',linewidth=4.0)
# plt.title('CGM_Entropy vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_Entropy')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #RMS
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_RMS'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_RMS'], color = 'blue',linewidth=4.0)
# plt.title('CGM_RMS vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_RMS')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #Correlation
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_Correlation'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_Correlation'], color = 'blue',linewidth=4.0)
# plt.title('CGM_Correlation vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_Correlation')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #Peaks
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_Peaks'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_Peaks'], color = 'blue',linewidth=4.0)
# plt.title('CGM_Peaks vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_Peaks')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #CGM_MinMax
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_MinMax'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_MinMax'], color = 'blue',linewidth=4.0)
# plt.title('CGM_MinMax vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_MinMax')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #CGM_Skewness
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_Skewness'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_Skewness'], color = 'blue',linewidth=4.0)
# plt.title('CGM_Skewness vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_Skewness')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #CGM_Displacement
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_Displacement'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_Displacement'], color = 'blue',linewidth=4.0)
# plt.title('CGM_Displacement vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_Displacement')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #CGM_Kurtosis
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_Kurtosis'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_Kurtosis'], color = 'blue',linewidth=4.0)
# plt.title('CGM_Kurtosis vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_Kurtosis')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #CGM_Recur
# plt.scatter(feat_dataset['CGM_days'], feat_dataset['CGM_Recur'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], feat_dataset['CGM_Recur'], color = 'blue',linewidth=4.0)
# plt.title('CGM_Recur vs Days')
# plt.xlabel('Days')
# plt.ylabel('CGM_Recur')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.rcParams['figure.dpi'] = 100
# plt.grid(True)
# plt.show()

# #########   PLOT   FOR    PC's    ###########
# ##PC1
# plt.scatter(feat_dataset['CGM_days'], pc['PC1'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], pc['PC1'], color = 'blue',linewidth=4.0)
# plt.title('PC1 vs Days')
# plt.xlabel('Days')
# plt.ylabel('PC1')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.grid(True)
# plt.show()

# ##PC2
# plt.scatter(feat_dataset['CGM_days'], pc['PC2'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], pc['PC2'], color = 'blue',linewidth=4.0)
# plt.title('PC2 vs Days')
# plt.xlabel('Days')
# plt.ylabel('PC2')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.grid(True)
# plt.show()

# ##PC3
# plt.scatter(feat_dataset['CGM_days'], pc['PC3'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], pc['PC3'], color = 'blue',linewidth=4.0)
# plt.title('PC3 vs Days')
# plt.xlabel('Days')
# plt.ylabel('PC3')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.grid(True)
# plt.show()


# ##PC4
# plt.scatter(feat_dataset['CGM_days'], pc['PC4'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], pc['PC4'], color = 'blue',linewidth=4.0)
# plt.title('PC4 vs Days')
# plt.xlabel('Days')
# plt.ylabel('PC4')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.grid(True)
# plt.show()

# ##PC5
# plt.scatter(feat_dataset['CGM_days'], pc['PC5'], color = 'red',linewidth=4.5)
# plt.plot(feat_dataset['CGM_days'], pc['PC5'], color = 'blue',linewidth=4.0)
# plt.title('PC5 vs Days')
# plt.xlabel('Days')
# plt.ylabel('PC5')
# plt.rcParams['figure.figsize']=(50,25)
# plt.rcParams.update({'font.size': 60})
# plt.grid(True)
# plt.show()
