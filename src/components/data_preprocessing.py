import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
class DataPreprocessing:
    def Preprocessdata(self,data):

        X=data.drop(['Loan_ID','Loan_Status'],axis=1)
        y=data['Loan_Status'].map({'N':0,'Y':1})

        cat_cols=X.select_dtypes(include='object').columns
        num_cols=X.select_dtypes(exclude='object').columns

        num_pipeline=Pipeline([
            ('missing_values',SimpleImputer(strategy='median')),
            ('scaling',StandardScaler()),
        ])

        cat_pipeline=Pipeline([
        ('missing',SimpleImputer(strategy='most_frequent')),
        ('encode',OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor=ColumnTransformer([
            ('num_cols',num_pipeline,num_cols),
            ('cat_cols',cat_pipeline,cat_cols)
        ])
        return X,y,preprocessor