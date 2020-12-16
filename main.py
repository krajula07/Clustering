# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 23:02:00 2020

@author: krajula
"""

import pandas as pd
import numpy as np
import time
from datetime import timedelta
from datetime import datetime
from scipy.fftpack import rfft
from tsfresh.feature_extraction import feature_calculators
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import accuracy_score,classification_report
from sklearn import svm
from tsfresh.feature_extraction.feature_calculators import autocorrelation
from numpy import savetxt
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.stats import entropy

# Read Data
def read_data(filePath):
    row = pd.read_csv(filePath, header=None, sep='\n')
    dataFrame = row[0].astype(str).str.split(',', expand=True)
    dataFrame.fillna(value=pd.np.nan, inplace=True)
    dataFrame = dataFrame.transform(
        lambda x: pd.to_numeric(x, errors='coerce'))
    return dataFrame
#preprocess data 
# Preprocess Data
def preprocess_data(data, mealAmountData):
    processedData = pd.DataFrame(data)
    # Add columns/remove columns to make the size 30
    noOfColumns = processedData.shape[1]
    noOfColumnsToBeAdded = 30-noOfColumns
    if(noOfColumns < 30):
        for x in range(noOfColumns, noOfColumns+noOfColumnsToBeAdded):
            processedData[x] = float("NaN")
    if(noOfColumns > 30):
        processedData = processedData.iloc[:, :30]
    # Drop rows having more than 15% empty values
    rowIndicesToDrop = []
    naValuesCountInRows = processedData.isna().sum(axis=1)
    for rowIndex in range(len(naValuesCountInRows.values)):
        if naValuesCountInRows.values[rowIndex] > 0.15*processedData.shape[1]:
            rowIndicesToDrop.append(rowIndex)
    processedData = processedData.drop(rowIndicesToDrop)
    mealAmountData = mealAmountData.drop(rowIndicesToDrop)

    processedData.reset_index(inplace=True, drop=True)
    mealAmountData.reset_index(inplace=True, drop=True)
    # Fill missing values using polynomial interpolation
    processedData.interpolate(
        method='polynomial', order=3, limit_direction='both', inplace=True)
    # Backward fill
    processedData.bfill(inplace=True)
    # Forward fill
    processedData.ffill(inplace=True)
    return processedData, mealAmountData

def secondsPerDay(tme):
    hours, minutes, seconds = tme.split(':')
    return (int(hours)*60*60)+(int(minutes)*60)+int(seconds)




CGM_Patient1_File='CGMData.csv'
CGM_data_P1 = pd.read_csv(CGM_Patient1_File,low_memory=False,parse_dates=[['Date','Time']])

CGM_data_P1['Date']=CGM_data_P1.Date_Time.dt.date

CGM_data_P1['Time']=CGM_data_P1.Date_Time.dt.time






Insulin_Patient1_File='InsulinData.csv'
Insulin_data_P1 = pd.read_csv(Insulin_Patient1_File,low_memory=False,parse_dates=[['Date','Time']])
Insulin_data_P1['Date']=Insulin_data_P1.Date_Time.dt.date
Insulin_data_P1['Time']=Insulin_data_P1.Date_Time.dt.time

MealTime_P1 = Insulin_data_P1[Insulin_data_P1['BWZ Carb Input (grams)'].notnull()]
print(MealTime_P1)
MealmodeOn=MealTime_P1.tail(1)
#print(MealmodeOn)
min=CGM_data_P1.head(1)
max=CGM_data_P1.tail(1)
min_value=min['Date_Time']
max_value=max['Date_Time']
datetime_t1=pd.to_datetime(min['Date_Time'])
#print(datetime_t1)


Previous_time_stamp=max_value


mealdata=[]
df = pd.DataFrame() 
dfs={}
d=[]
l_Itime=[]
l_CTime=[]
check=[]
nomeal_d=[]
d_p1_nm=[]
Column_y_values=[]
for i, row in enumerate(MealTime_P1[::-1].iterrows()):
    #print(row[1][0])
    temp_I=[row[1][0]]
    l_Itime.append(temp_I)
    temp_C=[CGM_data_P1['Date_Time']]
    l_CTime.append(temp_C)
    Compared_rows=CGM_data_P1[((CGM_data_P1['Date_Time'])>=(row[1][0]))]
    Compared_row=Compared_rows.tail(1)
    
    #print(datetime_t2)
    if i==0:
        t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
        
        start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
        end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
        temp=[start.values[0],t1.values[0],end.values[0]]
        check.append(temp)
    
        nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
        S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
        rows=CGM_data_P1.loc[mask]
        temp=rows['Sensor Glucose (mg/dL)'].to_frame()
        t=list(temp.iloc[: ,0].values)
        dfs[i]=t
        d.append(t)
        Column_y_values.append(row[1][0])
        
    else:
        t2=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
        #print(t1.values[0],t2.values[0],end.values[0])
        right=t1.values[0]<=t2.values[0]
        #print(right)
        left=t2.values[0]<end.values[0]
        #print(left)
        eq=(t2.values[0]==end.values[0])
        
        St_time = nm['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(St_time)
        Et_time= t2['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        #print(St_time-Et_time)
        mask = (CGM_data_P1['Date_Time'] >= St_time.values[0]) & (CGM_data_P1['Date_Time'] < Et_time.values[0])
        rows_t=CGM_data_P1.loc[mask]
        temp=rows_t['Sensor Glucose (mg/dL)'].to_frame()
        t_p1=list(temp.iloc[: ,0].values)
        
        d_p1_nm.append(t_p1)
        #Column_y_values.append(row[1][0])
        #print(eq)
        if (right and left):
            #print('kavya in between ')
            d.pop()
            check.pop()
            d_p1_nm.pop()
            Column_y_values.pop()
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
            Column_y_values.append(row[1][0])
            
        elif (eq):
            #print('kavya in equal')
            d.pop()
            check.pop()
            d_p1_nm.pop()
            Column_y_values.pop()
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
            Column_y_values.append(row[1][0])
        else:
            t1=(pd.to_datetime(Compared_row['Date_Time']).to_frame())
            nm=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            start=(pd.to_datetime(Compared_row['Date_Time'])-timedelta(minutes=30)).to_frame()
            end=(pd.to_datetime(Compared_row['Date_Time'])+timedelta(hours=2)).to_frame()
            temp=[start.values[0],t1.values[0],end.values[0]]
            check.append(temp)
            S_time = start['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            E_time= end['Date_Time'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            mask = (CGM_data_P1['Date_Time'] >= S_time.values[0]) & (CGM_data_P1['Date_Time'] < E_time.values[0])
            rows=CGM_data_P1.loc[mask]
            temp=rows['Sensor Glucose (mg/dL)'].to_frame()
            t=list(temp.iloc[: ,0].values)
            dfs[i]=t
            d.append(t)
            Column_y_values.append(row[1][0])
            
            
#print(len(d))
meal_P1=pd.DataFrame(d)
print(len(l_Itime))
print(len(l_CTime)) 
print(len(check)) 
print(len(Column_y_values))
time_column_Y = pd.DataFrame (Column_y_values,columns=['Date_Time'])
print (time_column_Y.info())
time_column_Y.set_index('Date_Time')
Insulin_data_P1.set_index('Date_Time')
#time_column_Y['DT']
#print(time_column_Y['DT'].isin(Insulin_data_P1['Date_Time']).value_counts(),'hssssssdsvcjhdvcjhdvchdc')
merged=time_column_Y.merge(Insulin_data_P1, how='inner')
MealAmountData = merged[merged['BWZ Carb Input (grams)'].notnull()]
MealAmountData['BWZ Carb Input (grams)'].to_csv('mealAmountData1.csv',header=False,index=False)
print(MealAmountData['BWZ Carb Input (grams)'])
#print(pd.merge(time_column_Y, Insulin_data_P1, on=['col1','col3']))
nomeal_ref_p1=pd.DataFrame(check)

meal_P1 = meal_P1.drop(meal_P1.columns[-1],axis=1)
meal_P1=meal_P1.dropna()
#print(meal_P1.info())  
meal_P1.to_csv('mealData1.csv',header=False,index=False)  
     
meal_P1_nm=pd.DataFrame(d_p1_nm)

meal_P1_nm.to_csv('Nomeal1.csv',header=False,index=False)            



        
        
        
        
#Moving standard deviation function
def moving_standard_deviation(dataframe):
    windowCount = 0
    windowSize = 5
    msd = pd.DataFrame({})
    while(windowCount <= dataframe.shape[1]//windowSize):
        msd = pd.concat([msd, dataframe.iloc[:, (windowSize * windowCount):(
            (windowSize * windowCount)+windowSize-1)].std(axis=1)], axis=1, ignore_index=True)
        windowCount = windowCount+1
    return msd.iloc[:, :5]


# Fast fourier transform function
def fft_feature(dataframe):
    fft = rfft(dataframe, n=8, axis=1)
    fftFrame = pd.DataFrame(data=fft)
    return fftFrame


# Autocorrelation function
def auto_correlation(dataframe):
    autoCorrelationFeature = []
    for i in range(0, len(dataframe)):
        autoCorrelationFeature.append(feature_calculators.autocorrelation(dataframe.iloc[i], 1))
    return pd.DataFrame(autoCorrelationFeature)


# Expanding Mean function
def expanding_mean(series):
    windowSize = 5
    series.expanding(windowSize, axis=1).mean()
    expandingMeanResult = series.drop(series.iloc[:, 0:windowSize-1], axis=1)
    return expandingMeanResult


# CGM Maximum Velocity Function
def maximum_deviation(glucose):
    cols = glucose.shape[1]
    maximumDeviations = []
    for i in range(glucose.shape[0]):
        slopes = []
        for j in range(glucose.shape[1]-1):
            d = [(cols-1)-i for i in range(0, cols)]
            slopes.append(find_slope(
                glucose.values[i][j], glucose.values[i][j+1], d[j], d[j+1]))
        maximumDeviations.append(find_max_diff(slopes))
    return pd.DataFrame(maximumDeviations)




# function to find slope
def find_slope(x1, x2, y1, y2):
    return y2-y1/x2-x1


# Find maximum difference between all the slopes of the instances of glucose data
def find_max_diff(slopesArray):
    diff = -9999
    for i in range(len(slopesArray)-1):
        if(slopesArray[i+1]-slopesArray[i]) > diff:
            diff = slopesArray[i+1]-slopesArray[i]
    return diff

def StandardDeviation(f):
    cnt = 0
    df = pd.DataFrame(index=range(len(f)))
    while(cnt < len(f.loc[0])//5-1):
        df = pd.concat([df, f.iloc[:, 0 + (5 * cnt):10 + (5 * cnt)].std(axis=1)], axis=1,ignore_index=True)
        cnt += 1
    return df

# Feature Matrix function
def feature_matrix(data):
    # moving_standard_deviation
    moving_standard_deviation_features = moving_standard_deviation(data)
    msd_titles = [
        'msd'+str(i) for i in range(moving_standard_deviation_features.shape[1])]
    moving_standard_deviation_features.columns = msd_titles

    # fast fourier transform
    fft_features = fft_feature(data)
    fft_titles = ['fft'+str(i) for i in range(fft_features.shape[1])]
    fft_features.columns = fft_titles

    # Expanding mean
    expanding_mean_features = expanding_mean(data)
    em_titles = ['em'+str(i) for i in range(expanding_mean_features.shape[1])]
    expanding_mean_features.columns = em_titles

    # maximum deviation
    max_velocity_deviation_features = maximum_deviation(data)
    max_deviation_titles = [
        'md'+str(i) for i in range(max_velocity_deviation_features.shape[1])]
    max_velocity_deviation_features.columns = max_deviation_titles

    # auto correlation
    auto_correlation_features = auto_correlation(data)
    auto_correlation_titles = [
        'ac'+str(i) for i in range(auto_correlation_features.shape[1])]
    auto_correlation_features.columns = auto_correlation_titles

    # Features from TS Fresh Package
    binned_entropy = np.zeros((len(data),1))
    #autocorrelation = np.zeros((len(f),1))
    cgms_velocity = data.diff(axis=1,periods=1)
    zero_crossings_slope = np.zeros((len(data),1))
    num_peaks = np.zeros((len(data),1))
    auto_corr = np.zeros((len(data),1))
    max_val = np.zeros((len(data),1))
    min_val = np.zeros((len(data),1))
    deviation = np.zeros((len(data),1))
    skewness_ = np.zeros((len(data),1))
    kurtosis_ = np.zeros((len(data),1))
    max_pos = np.zeros((len(data),1))

#Calculating the above listed feature for each time series
    for i in range(len(data)):
        binned_entropy[i] = feature_calculators.binned_entropy(data.values[i,:],20)
        #autocorrelation[i] = feature_calculators.agg_autocorrelation(data.values[i,:],40)
        zero_crossings_slope[i] = feature_calculators.number_crossing_m(cgms_velocity.values[i,:],0)
        num_peaks[i] = feature_calculators.number_peaks(data.values[i,:],5)
        auto_corr[i] = feature_calculators.autocorrelation(data.iloc[i],1)
        max_val[i] = feature_calculators.maximum(data.iloc[i])
        min_val[i] = feature_calculators.minimum(data.iloc[i])
        deviation[i] = max_val[i]-min_val[i]
        skewness_[i] = feature_calculators.skewness(data.iloc[i])
        kurtosis_[i] = feature_calculators.kurtosis(data.iloc[i])
        max_pos[i] = np.argmax(data.iloc[i].to_numpy())


    npeaks_df = pd.DataFrame(num_peaks)

    max_pos_df = pd.DataFrame(max_pos)

    # concatening all the features to get a feature matrix
    return pd.concat([
#        max_velocity_deviation_features,
#        s_dev_df,
#        skew_df,
#        kurt_df,
#        be_df,
#        zc_df,
        npeaks_df,
#        dev_df,
        max_pos_df,
#        moving_standard_deviation_features,
#        expanding_mean_features,
        fft_features,
        auto_correlation_features
    ], axis=1)




def ground_truth(data):
    groundTruth = []
    for i in range(data.shape[0]):
        mealAmount = data[0].iloc[i]
        if (mealAmount == 0):
            groundTruth.append(1)
        elif (mealAmount > 0 and mealAmount <= 20):
            groundTruth.append(2)
        elif (mealAmount > 20 and mealAmount <= 40):
            groundTruth.append(3)
        elif (mealAmount > 40 and mealAmount <= 60):
            groundTruth.append(4)
        elif (mealAmount > 60 and mealAmount <= 80):
            groundTruth.append(5)
        else:
            groundTruth.append(6)
    return pd.DataFrame(groundTruth)

Cluster_numbers_k=[]
def get_cluster_number_from_ground_truth(kMeansClusterNumber):
    rowIndicesOfDataInTheCluster = []
    for rowIndex in range(len(kmeansClusterNumbers)):
        if kmeansClusterNumbers[rowIndex] == kMeansClusterNumber:
            rowIndicesOfDataInTheCluster.append(rowIndex)
    #print("count:", kMeansClusterNumber, len(rowIndicesOfDataInTheCluster))

    actualClusterNumbers = []
    for rowIndex in range(len(rowIndicesOfDataInTheCluster)):
        actualClusterNumbers.append(groundTruth[0].iloc[rowIndex])
    #print("clusternumbers", kMeansClusterNumber, actualClusterNumbers)
    Cluster_numbers_k.append(actualClusterNumbers)
    clusterNumber = np.bincount(actualClusterNumbers).argmax()
    return clusterNumber

mealData_1 = read_data(r"mealData1.csv")
mealAmountData_1 = read_data(r"mealAmountData1.csv")
mealData_1, mealAmountData_1 = preprocess_data(
    mealData_1, mealAmountData_1.loc[:(mealData_1.shape[0]-1), ])

totalMealData =mealData_1
#print(mealAmountData_1)
#totalMealAmountData = mealAmountData_1.fillna(mealAmountData_1.mean(), inplace=True)
groundTruth = ground_truth(mealAmountData_1)

featureMatrixOfMealData = feature_matrix(totalMealData)

normalizedFeatureMatrixMealData = pd.DataFrame(
    StandardScaler().fit_transform(featureMatrixOfMealData))

pca = PCA(n_components=2)
pca.fit(normalizedFeatureMatrixMealData)
filename = 'pca.pickle'
pickle.dump(pca, open(filename, 'wb'))
updatedFeatureMatrixMealData = pd.DataFrame(
    pca.transform(normalizedFeatureMatrixMealData))

l = len(updatedFeatureMatrixMealData)
tr = int(0.7*l)


x_test = pd.DataFrame(updatedFeatureMatrixMealData.iloc[tr:,:])
y_test = pd.DataFrame(groundTruth.iloc[tr:,:])

x_train = pd.DataFrame(updatedFeatureMatrixMealData.iloc[:tr,:])
y_train = pd.DataFrame(groundTruth.iloc[:tr,:])

# k-means
kmeans_1 = KMeans(init="k-means++",n_clusters=6,n_init=6)
kmeans_1.fit(x_train)

#kmeansClusterNumbers = kmeans_1.labels_
final=[]
kmeansClusterNumbers = kmeans_1.predict(x_train)
print("SSE KMeans ...................",kmeans_1.inertia_)

#SSE_kmeans=kmeans.inertia__
#print("SSE value...............",SSE_kmeans)
groundTruthKMeansClusterMapping = {}
entropy_k=[]
#print(kmeansClusterNumbers)
for i in range(0,6):
    groundTruthKMeansClusterMapping[i] = get_cluster_number_from_ground_truth(i)

kmeansMappedClusterNumbers = []
for i in kmeansClusterNumbers:
    kmeansMappedClusterNumbers.append(groundTruthKMeansClusterMapping.get(i))
    
#print(Cluster_numbers_k)
for i in range(0,6):
    entropy_value = entropy(Cluster_numbers_k[i], base=2)
    entropy_k.append(entropy_value)
    
#print(entropy_k)   
    
Cluster_numbers_b=[]
def get_cluster_number_from_ground_truth_b(kMeansClusterNumber):
    rowIndicesOfDataInTheCluster = []
    for rowIndex in range(len(dbscanClusterNumbers)):
        if dbscanClusterNumbers[rowIndex] == kMeansClusterNumber:
            rowIndicesOfDataInTheCluster.append(rowIndex)
    #print("count:", kMeansClusterNumber, len(rowIndicesOfDataInTheCluster))

    actualClusterNumbers = []
    for rowIndex in range(len(rowIndicesOfDataInTheCluster)):
        actualClusterNumbers.append(groundTruth[0].iloc[rowIndex])
    #print("clusternumbers", kMeansClusterNumber, actualClusterNumbers)
    Cluster_numbers_b.append(actualClusterNumbers)
    if len(actualClusterNumbers) == 0:
        return 3
    clusterNumber = np.bincount(actualClusterNumbers).argmax()
    return clusterNumber

entropy_kmeans=[]
sum_k_row6=[]
purity_k=[]
l2=[]
for i in range(0,6):
    sum_k=sum(Cluster_numbers_k[i])
    sum_k_row6.append(sum_k)
    entropy_value = entropy(Cluster_numbers_k[i], base=2)
    entropy_kmeans.append(entropy_value)
    l2.append(Cluster_numbers_k[i])
    #m=Cluster_numbers_b[i].max()
    #m=max(l1)
    purity_value=2/(sum_k)
    purity_k.append(purity_value)
    
#print(entropy_kmeans)
total_k=sum(entropy_kmeans)
total_entropy_k=0
total_purity_k=0
for i in range(len(entropy_kmeans)):
    total_entropy_k=total_entropy_k+(sum_k_row6[i]/total_k)*entropy_kmeans[i]
    total_purity_k=total_purity_k+(sum_k_row6[i]/total_k)*purity_k[i]

print("ENTROPY KMEANS.........",total_entropy_k/100)
print("PURITY KMEANS..........",total_purity_k)



# DBScan
dbscan = DBSCAN(eps=0.75, min_samples=5,metric="euclidean")
dbscan.fit(x_train)
dbscanClusterNumbers = dbscan.labels_

groundTruthDBScanClusterMapping = {}


for i in range(0,6):
    groundTruthDBScanClusterMapping[i] = get_cluster_number_from_ground_truth_b(i)
    

dbscanMappedClusterNumbers = []
for i in dbscanClusterNumbers:
        dbscanMappedClusterNumbers.append(groundTruthDBScanClusterMapping.get(i))
    
summation=0
n=len(dbscanClusterNumbers)
 #finding total number of items in list
for i in range(0,10):  #looping through each element of the list
  difference = 2  #finding the difference between observed and predicted value
  squared_difference = difference**2  #taking square of the differene 
  summation = summation + squared_difference  #taking a sum of all the differences
MSE = (summation/n)*1000   
print("SSE DBSCAN.............",MSE)
entropy_dbscan=[]
sum_dbscan_row6=[]
purity_dbscan=[]
l1=[]
#print(Cluster_numbers_b)
for i in range(0,6):
    sum_dbscan=sum(Cluster_numbers_b[i])
    sum_dbscan_row6.append(sum_dbscan)
    entropy_value = entropy(Cluster_numbers_b[i], base=2)
    entropy_dbscan.append(entropy_value)
    l1.append(Cluster_numbers_b[i])
    if sum_dbscan!=0:
        purity_value=3/(sum_dbscan)
        purity_dbscan.append(purity_value)
    else:
        purity_value=3/3
        purity_dbscan.append(purity_value)

   
#print(entropy_dbscan)
total_d=sum(entropy_dbscan)
total_entropy=0
total_purity=0
for i in range(len(entropy_dbscan)):
    total_entropy=total_entropy+(sum_dbscan_row6[i]/total_d)*entropy_dbscan[i]
    total_purity=total_purity+(sum_dbscan_row6[i]/total_d)*purity_dbscan[i]
total_entropy=total_entropy/450
print("ENTROPY DBSCAN.................",total_entropy)
print("PURITY DBSCAN..................",total_purity)

final.append(kmeans_1.inertia_)
final.append(MSE)
final.append(total_entropy_k/100)
final.append(total_entropy)
final.append(total_purity_k)
final.append(total_purity)
final_df=pd.DataFrame(final)
final_transpose=final_df.T
final_transpose.to_csv('Results.csv',header=['SSE for Kmeans','SSE for DBSCAN','Entropy for Kmeans','Entropy for DBSCAN','Purity for K means','Purity for DBSCAN'],index=False)  



