# ====   PATHS ===================

TRAINING_DATA_FILE = "titanic.csv"
PIPELINE_NAME = 'logistic_regression'


# ======= FEATURE GROUPS =============

TARGET = 'survived'

CATEGORICAL_VARS = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare']

extract_number_from_cabin = ['cabin']