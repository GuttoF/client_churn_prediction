import pickle
import inflection
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 42

def loading_data(df):
    """
    Load data from csv file and convert to pickle.
    input: [df] - dataframe
    return: [df] - dataframe with modifications
    """

    seed = 42

    # Loading data
    homepath = '/home/gutto/repos/github/client_churn_prediction/'
    df = pd.read_csv(homepath + 'data/raw/churn.csv')
    
    # Converting in pickle
    df.to_pickle(homepath + 'data/interim/churn.pkl')
    df = pd.read_pickle(homepath + 'data/interim/churn.pkl')

    # Dropping columns
    df.drop(columns = ['RowNumber', 'Surname', 'HasCrCard'], inplace = True)

    # Snake case
    cols_old = df.columns
    snake_case = lambda x: inflection.underscore(x)
    cols_new = list(map(snake_case, cols_old))
    df.columns = cols_new
    
    # Lower
    df[['gender', 'geography']] = df[['gender', 'geography']].apply(lambda x: x.str.lower())

    return df


def feature_engineering(df):
    """

    """







cols_drop = ['estimated_salary_per_country', 'tenure_per_country', 'credit_score_per_gender', '_has_cr_card']



