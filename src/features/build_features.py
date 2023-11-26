import pandas as pd
from typing import Union



def build_features(X_train: Union[int, float, str], X_test: Union[int, float, str], y_train: Union[int, float, str], y_test: Union[int, float, str], id_train: Union[int, float, str], id_test: Union[int, float, str]):
    """
    Build features for the model.
    input: [X_train, X_test, y_train, y_test, id_train, id_test, X_val, y_val] - dataframes
    return: [X_train, X_test, y_train, y_test, id_train, id_test, X_val, y_val] - dataframes with modifications
    """

    for dataframe in [X_train, X_test]:
        # balance_salary_ratio
        dataframe['balance_salary_ratio'] = dataframe['balance']/dataframe['estimated_salary']
        # credit_score_age_ratio
        dataframe['credit_score_age_ratio'] = dataframe['credit_score']/dataframe['age']
        # tenure_age_ratio
        dataframe['tenure_age_ratio'] = dataframe['tenure']/dataframe['age']
        # life_stage
        dataframe['life_stage'] = dataframe['age'].apply(lambda x: 'adolescence' if x <= 20 else 'adulthood' if (
            x > 20) & (x <= 35) else 'middle_age' if (x > 35) & (x <= 50) else 'senior')

    # balance_age
    balance_age_train, balance_age_test, balance_age_val = [dataframe.loc[:, ['age', 'balance']].groupby('age').mean().reset_index() for dataframe in [X_train, X_test]]

    balance_age_train.columns = ['age', 'balance_per_age']
    balance_age_test.columns = ['age', 'balance_per_age']

    X_train = pd.merge(X_train, balance_age_train, on = 'age', how = 'left')
    X_test = pd.merge(X_test, balance_age_test, on = 'age', how = 'left')

    # LTV
    balance_tenure_train, balance_tenure_test = [dataframe.loc[:, ['tenure', 'balance']].groupby('tenure').mean().reset_index() for dataframe in [X_train, X_test]]

    balance_tenure_train.columns = ['tenure', 'ltv']
    balance_tenure_test.columns = ['tenure', 'ltv']

    X_train = pd.merge(X_train, balance_tenure_train, on = 'tenure', how = 'left')
    X_test = pd.merge(X_test, balance_tenure_test, on = 'tenure', how = 'left')

    return X_train, X_test, y_train, y_test, id_train, id_test