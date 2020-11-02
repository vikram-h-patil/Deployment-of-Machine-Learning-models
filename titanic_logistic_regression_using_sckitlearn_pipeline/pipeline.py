from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


titanic_pipe = Pipeline(
    [
        ('extract_first_letter',
            pp.ExtractFirstLetter(variables=config.extract_number_from_cabin)),
        ('add_na_binary_variable',
            pp.MissingIndicator(variables=config.NUMERICAL_VARS)),
        ('impute_numerical_variable',
             pp.NumericalImputer(variables=config.NUMERICAL_VARS)),
        ('impute_categorical_missing',
             pp.CategoricalImputer(variables=config.CATEGORICAL_VARS)),
        ('rare_categorical_encoder',
             pp.RareLabelCategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('onehot_categorical_encoder',
             pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
        ('scaler', StandardScaler()),
        ('Logistic_model', LogisticRegression(C=0.0005, random_state=0))
         
    ]
)