# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 21:27:11 2021

@author: PriyankRao
"""
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime  as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor as varfac
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from seaborn import kdeplot
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from math import sqrt


os.chdir(r"C:\Users\PriyankRao\OneDrive - E2\Documents\Project\webscrapping")



class housingPrice():
    def dataCleaning(self, df):
        print("***filling na with mean values****")
        mean_lot = df['Lot Area'].mean()
        df['Lot Area'].fillna(value = mean_lot, inplace =  True)

        mean_bed = df['Bed'].mean()
        df['Bed'].fillna(value = mean_bed, inplace = True)

        mean_bath = df['Bath'].mean()
        df['Bath'].fillna(value = mean_bath, inplace = True)
        
        mean_liv = df['living Area'].mean()
        df['living Area'].fillna(value = mean_liv, inplace = True)
        
        index = list(df[df['living Area']>=10000].index)
        df = df.drop(index = index, axis = 0)
        df = df.reset_index(drop = True)
        index = list(df[df['Lot Area']==df['Lot Area'].max()].index)
        df=df.drop(index = index, axis= 0)
        df= df[~df['House Type'].str.contains('MANUFACTURED')]
        df= df[~df['House Type'].str.contains('LOT')]
        
        print("Removed the outlier and accessing the scatterplot (1):")
        df['Difference'] = df['Bed'] - df['Bath']
        variables = df[['Difference', 'House Type', 'Lot Area', 'Bath', 'Bed', 'living Area', 'Price']]
        sns_plot = sns.pairplot(variables,  height=2.5)
        
        return df
        
     
    def VIF(self, df):
        print("Working to get the VIF factor between indepedent variables(1):.....")
        variables = df[[ 'Difference', 'living Area', 'Lot Area']]
        hpfeature = variables
        vif_data = pd.DataFrame()
        vif_data["feature"] = variables.columns 
        vif_data["VIF"] = [varfac(hpfeature.values, i) for i in range(len(hpfeature.columns))]
        print(vif_data)
 
    def Clustering(self, df):
        
        df_ind = df[['Difference', 'House Type', 'Lot Area', 'Bath', 'Bed', 'living Area']]
        df_dep = df[['Price']]
        df_geo = df[['Latitude', 'Longitude']]
        X = df_geo[['Latitude', 'Longitude']].values
        Z = linkage(X,
                    method='complete',  # dissimilarity metric: max distance across all pairs of 
                                    # records between two clusters
                    metric='euclidean'
                    )                           # you can peek into the Z matrix to see how clusters are 
                                        # merged at each iteration of the algorithm

        # retrive clusters with `max_d`
        max_d = 1.3  # I assume that your `Latitude` and `Longitude` columns are both in 
                     # units of miles
        clusters = fcluster(Z, max_d, criterion='distance')
        df_geo['Clusters'] = clusters.tolist()
        values ={0:'A', 1:'B', 2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N', 14:'O', 15:'P',
                 16:'Q', 17:'R',18:'S', 19:'T',20:'U', 21:'V',22:'W', 23:'X',24:'Y',25:'Z'    }
        df_geo = df_geo.replace({'Clusters': values})
        df_geo.to_excel(r"geo.xlsx", index = False)
        df_ivar = df_ind.join(df_geo)
        hp = df_ivar.join(df_dep)
        hp = hp.drop(columns = ['Latitude', 'Longitude'], axis = 1)
        
        return hp
   
    def LinearRegression(self, hp):
        enc_df = pd.get_dummies(hp, columns =['House Type', 'Clusters'])
         
        X = enc_df.drop(['Price'], axis=1)
        X = X[['living Area',
       'House Type_APARTMENT', 'House Type_CONDO', 'House Type_MULTI_FAMILY',
       'House Type_SINGLE_FAMILY', 'House Type_TOWNHOUSE', 'Clusters_B',
       'Clusters_C', 'Clusters_D', 'Clusters_E', 'Clusters_F', 'Clusters_G',
       'Clusters_H', 'Clusters_I', 'Clusters_J', 'Clusters_K', 'Clusters_L',
       'Clusters_M', 'Clusters_N', 'Clusters_O', 'Clusters_P', 'Clusters_Q',
       'Clusters_R', 'Clusters_S', 'Clusters_T', 'Clusters_U', 'Clusters_V']]
        print("Printing the independent variables......")
        print(X.columns)
        #, 'Lot Area'
        x = X[['living Area']]
        Y =enc_df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size =0.7, test_size = 0.3, random_state = 3)
        
        X_train_sm = sm.add_constant(X_train)
        
        lr = sm.OLS(y_train, X_train_sm).fit()
        print(lr.params)
        print(lr.summary())
        X_test_sm = sm.add_constant(X_test)
        y_test_pred = lr.predict(X_test_sm)
        pre = pd.DataFrame(y_test_pred, columns={'Prediction'})
        print("=============================================================================")        
        
        rmse = round(sqrt(mean_squared_error(y_test, y_test_pred)),2)  
        print("The RMSE for the regression model(OLS) is: ", rmse)
        print("=============================================================================")        

        dataframe =  pd.merge(enc_df, X_test_sm, left_index = True, right_index = True, how = 'right')
        dataframe = pd.merge(dataframe, y_test, left_index = True, right_index = True, how = 'right')
        dataframe = pd.merge(dataframe, pre, left_index = True, right_index = True, how = 'right')
        dataframe.to_excel('LR Prediction.xlsx', index = False)
        
    def Ridge(self, hp):
        enc_df = pd.get_dummies(hp, columns =['House Type', 'Clusters'])
         
        X = enc_df.drop(['Price'], axis=1)
        X = X[['living Area',
       'House Type_APARTMENT', 'House Type_CONDO', 'House Type_MULTI_FAMILY',
       'House Type_SINGLE_FAMILY', 'House Type_TOWNHOUSE', 'Clusters_B',
       'Clusters_C', 'Clusters_D', 'Clusters_E', 'Clusters_F', 'Clusters_G',
       'Clusters_H', 'Clusters_I', 'Clusters_J', 'Clusters_K', 'Clusters_L',
       'Clusters_M', 'Clusters_N', 'Clusters_O', 'Clusters_P', 'Clusters_Q',
       'Clusters_R', 'Clusters_S', 'Clusters_T', 'Clusters_U', 'Clusters_V']]
        
        print("Working on ridge regression......")
        
        #, 'Lot Area'
        x = X[['living Area']]
        Y =enc_df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size =0.7, test_size = 0.3, random_state = 3)
        
        rr = Ridge(alpha = 10)
        rr.fit(X_train, y_train)
        Ridge_train_score = rr.score(X_train, y_train)
        Ridge_test_score = rr.score(X_test, y_test)
        
        print("Getting the prediction via Ridge regression......")
        prediction = rr.predict(X_test)
        pred_df = pd.DataFrame(prediction, columns = ['Prediction'])
        X_test.columns
        ind_sample = y_test
        ind_sample = ind_sample.reset_index(drop =True)
        pred_df = pred_df.join(ind_sample)
        pred_df = pred_df.round(2)
        
        print("=============================================================================")        
        rmse = round(sqrt(mean_squared_error(pred_df['Price'], pred_df['Prediction'])),2)  
        print("The root mean square error for the Ridge regression model is: ", rmse)
        print("=============================================================================")        
        
        
        # print("Ridge Train Score: ")
        # print(Ridge_train_score)
        
        # print("Ridge Test Score: ")
        # print(Ridge_test_score)
        
    def run(self):
        df = pd.read_csv(r'houseinfo.csv')
        df = df[['Latitude', 'Longitude', 'House Type','Lot Area', 'Lot Area Unit', 'Bath', 'Bed','living Area', 'Price']]
        df['Lot Area'] = np.where(df['Lot Area Unit'] == 'acres', df['Lot Area']*43560, df['Lot Area'])
        df = self.dataCleaning(df)
        self.VIF(df)
        hp = self.Clustering(df)
        self.LinearRegression(hp)
        self.Ridge(hp)
        
        
if __name__ == '__main__':
    pred = housingPrice()
    pred.run()