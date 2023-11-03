import pickle
from typing                             import Union
from sklearn.pipeline                   import Pipeline
from sklearn.preprocessing              import LabelEncoder
from sklearn.compose                    import ColumnTransformer

def pipeline_churn(dataframe: Union[int, float, str], min_max_scaler: list, mms: str, robust_scaler: list, rs: str, standard_scaler: list, ss: str, cols_ohe: list, ohe: str, cols_le: list, homepath: str, save_scalers: bool = True, scalers_filename: str = 'scalers.pkl'):
    """
    This function takes a Pandas DataFrame as input, along with three lists of column names (`min_max_scaler`, `robust_scaler`, `standard_scaler` `cols_ohe` and `cols_le`). The function applies a series of transformations to the input DataFrame, including one-hot encoding, label encoding, min-max scaling, standard scaling and robust scaling.

    Args:
        dataframe (Union[int, float, str]): Dataframe with all features
        min_max_scaler (list): List of Features to apply mms
        robust_scaler (list): List of Features to apply rs
        standard_scaler (list): List of Features to apply ss
        cols_ohe (list): List of Features to apply ohe
        cols_le (list): List of Features to apply le
        homepath (str): Path to save the scalers
        save_scaler (bool): Save scalers to a pickle file
        scalers_filename (str): Name of the pickle file

    Returns:
        dataframe: dataframe with transformations
    """

    le_dict = {}
    for col in cols_le:
        le_dict[col] = LabelEncoder()
        dataframe[col] = le_dict[col].fit_transform(dataframe[col])

    # column transformer
    column_transformer = ColumnTransformer(
        transformers = [('mms', mms, min_max_scaler),
                        ('rs', rs, robust_scaler),
                        ('ss', ss, standard_scaler),
                        ('ohe', ohe, cols_ohe)], remainder = 'passthrough')

    # pipeline
    pipeline = Pipeline([('column_transform', column_transformer)])
    dataframe = pipeline.fit_transform(dataframe)

    # Get column name
    column_names = column_transformer.get_feature_names_out()

    # Get dataframe back
    dataframe = pd.DataFrame(dataframe, columns = column_names)
    dataframe = dataframe.rename(columns = lambda x: x.replace('remainder_', ''))
    dataframe = dataframe.rename(columns = lambda x: x.replace('mms__', ''))
    dataframe = dataframe.rename(columns = lambda x: x.replace('rs__', ''))
    dataframe = dataframe.rename(columns = lambda x: x.replace('ss__', ''))
    dataframe = dataframe.rename(columns = lambda x: x.replace('ohe__', ''))
    dataframe = dataframe.rename(columns = lambda x: x.replace('le__', ''))

    # Store scalers objects
    global mms_scaler, rs_scaler, ss_scaler
    mms_scaler = column_transformer.named_transformers_['mms']
    rs_scaler = column_transformer.named_transformers_['rs']
    ss_scaler = column_transformer.named_transformers_['ss']

    # Store encoders objects
    global ohe_encoderm, le_encoder
    ohe_encoder = column_transformer.named_transformers_['ohe']
    le_encoder = le_dict

    # Save scalers to a pickle file
    if save_scalers == True:
        with open(homepath + scalers_filename, 'wb'):
                  pickle.dump({'mms_scaler': mms_scaler, 'rs_scaler': rs_scaler, 'ss_scaler': ss_scaler, 'ohe_encoder': ohe_encoder, 'le_encoder': le_encoder}, open(homepath + scalers_filename, 'wb'))
    
    return dataframe
