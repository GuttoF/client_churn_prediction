import inflection
import pandas

def loading_data(df):
    """
    Load data from csv file and convert to pickle.
    input: [df] - dataframe
    return: [df] - dataframe with modifications
    """

    # Dropping columns
    df.drop(columns=['RowNumber', 'Surname', 'HasCrCard'], inplace=True)

    # Convert column names to snake case
    cols_old = df.columns
    snake_case = lambda x: inflection.underscore(x)
    cols_new = list(map(snake_case, cols_old))
    df.columns = cols_new
    
    # Convert 'gender' and 'geography' columns to lowercase
    df[['gender', 'geography']] = df[['gender', 'geography']].apply(lambda x: x.str.lower())

    return df