from tqdm import tqdm   
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def timeseries_impute(df, pat_enc_id, column_list_to_impute): # ### Perform for the same patient with ffill followed by bfill 
    '''
    Inputs:
    df: pandas dataframe to be imputed (pandas.dataframe)
    pat_enc_id: name of encounter-id column of df (string)
    column_list_to_impute: name list of columns to be imputed (list) 
    '''
    column_list_not_to_impute = list(set(list(df.columns)) - set(column_list_to_impute))    
    df_imputed1 = pd.DataFrame([])
    for i in tqdm(df[pat_enc_id].unique()):
        df_i = df[df[pat_enc_id]==i]
        df_noImpute_i = df_i.loc[:, column_list_not_to_impute]
        df_toImpute_i = df_i.loc[:, column_list_to_impute]
        df_toImpute_i.fillna(method='ffill', inplace=True) 
        df_toImpute_i.fillna(method='bfill', inplace=True) 
        dfAll_i = pd.concat([df_noImpute_i, df_toImpute_i], axis=1)
        df_imputed1 = pd.concat([df_imputed1, dfAll_i], axis=0)
    #df_imputed1.loc[:,column_list_to_impute].isnull().sum().sum()
    return df_imputed1



def mice_numeric_impute(df, column_list_to_impute, verbose=2): #For continuous variables
    '''
    Inputs:
    df: pandas dataframe to be imputed (pandas.dataframe)
    column_list_to_impute: name list of columns to be imputed (list) 
    '''
    imp = IterativeImputer(max_iter=100, random_state=0, verbose=verbose)
    column_list_not_to_impute = list(set(list(df.columns)) - set(column_list_to_impute))
    df_noImpute = df.loc[:, column_list_not_to_impute]
    df_toImpute = df.loc[:,column_list_to_impute].copy()
    imp.fit(df_toImpute)

    df_impute = imp.transform(df_toImpute)
    df_impute = pd.DataFrame(df_impute, columns=column_list_to_impute)
    df_imputed1 = pd.concat([df_noImpute, df_impute], axis=1)
    return df_imputed1
    


    
def median_numeric_impute(df, column_list_to_impute):
    '''
    Inputs:
    df: pandas dataframe to be imputed (pandas.dataframe)
    column_list_to_impute: name list of columns to be imputed (list) 
    '''
    column_list_not_to_impute = list(set(list(df.columns)) - set(column_list_to_impute))
    df_noImpute = df.loc[:, column_list_not_to_impute]
    df_toImpute = df.loc[:,column_list_to_impute].copy()
    df_toImpute.fillna(df_toImpute.median(), inplace=True)
    df_imputed1 = pd.concat([df_noImpute, df_toImpute], axis=1)
    return df_imputed1
    
   


'''
Unsatisfied with MICE for whole mixed data
Instead MICE-binary, MICE-ordinal, MICE-linear are used now

'''

from sklearn.linear_model import LogisticRegression, BayesianRidge, Ridge
import random

# -------- STEP 1: Create seed --------
np.random.seed(42)

def mice_mixed_type_data_impute(df, column_list_to_impute):
    # -------- STEP 2: Define variable types --------
    binary_cols = ['qSOFA_sbp', 'qSOFA_rr', 'qSOFA_gcs']
    ordinal_cols = ['qSOFA_total']
    continuous_cols = [col for col in column_list_to_impute if col not in binary_cols + ordinal_cols]

    # -------- STEP 3: Custom ordinal regressor --------
    class OrdinalRegressor(Ridge):
        def predict(self, X):
            pred = super().predict(X)
            return np.clip(np.round(pred), 0, 3)

    # -------- STEP 4: Impute binary columns --------
    df_imputed = df.copy()
    for col in binary_cols:
        imputer_bin = IterativeImputer(
            estimator=LogisticRegression(solver='liblinear'),
            max_iter=60,
            initial_strategy='most_frequent',
            random_state=42,
            verbose=2
        )
        df_imputed[[col]] = imputer_bin.fit_transform(df[[col]])
        df_imputed[col] = df_imputed[col].round().astype(int)

    # -------- STEP 5: Impute ordinal column --------
    imputer_ord = IterativeImputer(
        estimator=OrdinalRegressor(),
        max_iter=60,
        initial_strategy='most_frequent',
        random_state=42,
        verbose=2
    )
    df_imputed[ordinal_cols] = imputer_ord.fit_transform(df[ordinal_cols])
    df_imputed[ordinal_cols] = df_imputed[ordinal_cols].round().astype(int)

    # -------- STEP 6: Impute continuous columns --------
    imputer_cont = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=60,
        initial_strategy='mean',
        random_state=42,
        verbose=2
    )
    df_imputed[continuous_cols] = imputer_cont.fit_transform(df[continuous_cols])

    return df_imputed
