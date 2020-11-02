import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


# Add binary variable to indicate missing values
class MissingIndicator(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables


    def fit(self, X, y=None):
         return self


    def transform(self, X):
        X = X.copy()
        # Identifying which columns have null values
        #self.variables = X[self.variables].columns[X[self.variables].isna().any()].tolist() # filter 'age' and 'fare'
        for var in self.variables:
            X[var+'_NA'] = np.where(X[var].isnull(), 1, 0)
        return X


# categorical missing value imputer
class CategoricalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        # we need the fit statement to accommodate the sklearn pipeline
         return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].fillna('Missing')
        return X



# Numerical missing value imputer
class NumericalImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        #self.variables = X[self.variables].columns[X[self.variables].isna().any()].tolist() # filter 'age' and 'fare'
        # persist mode in a dictionary
        self.imputer_dict_ = {}
        
        for var in self.variables:
            self.imputer_dict_[var]=X[var].mean()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self.imputer_dict_[var], inplace=True)
        return X

# Extract first letter from string variable
class ExtractFirstLetter(BaseEstimator, TransformerMixin):
    
    def __init__(self,variables=None):
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        X.copy()
        for var in self.variables:
            X[var] = X[var].str[0] # captures the first letter
        return X

# frequent label categorical encoder
class RareLabelCategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, tol=0.05, variables=None):
        self.variables = variables
        self.tol=tol

    def fit(self, X, y=None):
        self.encoder_dict_ = {}
        # persist frequent labels in dictionary
        for var in self.variables:
            rare_var = X.groupby(var)[var].count()/len(X)
            self.encoder_dict_[var] = rare_var[rare_var<=self.tol].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(self.encoder_dict_[var]), 'Rare',X[var])
        return X


# string to numbers categorical encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):

        # HINT: persist the dummy variables found in train set
        self.dummies = pd.get_dummies(X[self.variables], drop_first=True).columns
        
        return self

    def transform(self, X):
        # encode labels
        X = X.copy()
        
        # get dummies
        X = pd.concat([X,pd.get_dummies(X[self.variables], drop_first=True)], axis=1)
        
        # drop original variables
        X.drop(labels=self.variables, axis=1, inplace=True)
        
        # add missing dummies if any
        missing_cols = [each_col for each_col in self.dummies if each_col not in X.columns]
        
        if len(missing_cols) != 0:
            for each_col in missing_cols:
                X[each_col] = 0
            
        return X
