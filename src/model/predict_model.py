from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from typing import Union
import pandas as pd
import numpy as np
import argparse
import pickle


def load_model(homepath):
    """
    Load a model from a file.
    input: homepath - path to the project
    return: model - model object
    """
    model = CatBoostClassifier()
    model.load_model(homepath + '/models/model.cbm')
    
    return model


def preprocess_data(dataframe: Union[int, float, str], min_max_scaler: list, robust_scaler: list, ohe_cols: list, log_cols: list):
    """
    This function takes a Pandas DataFrame as input, along with three lists of column names (`min_max_scaler`, `robust_scaler`, `standard_scaler` and `cols_ohe`). The function applies a series of transformations to the input DataFrame, including one-hot encoding, label encoding, min-max scaling, standard scaling and robust scaling.

    Args:
        dataframe (Union[int, float, str]): Dataframe with all features
        min_max_scaler (list): List of Features to apply mms
        robust_scaler (list): List of Features to apply rs
        standard_scaler (list): List of Features to apply ss
        ohe_cols (list): List of Features to apply ohe
        log_cols (list): List of Features to apply log

    Returns:
        dataframe: dataframe with transformations
        preprocessor: preprocessor object
    """

    transformers = []

    # Apply log transformation to specified columns
    for col in log_cols:
        transformers.append((f'log_transform_{col}', FunctionTransformer(np.log1p, validate = False), [col]))

    # Define the column transformers for different types of scaling and encoding
    transformers += [('mm_scaler', MinMaxScaler(), min_max_scaler),
                     ('rb_scaler', RobustScaler(), robust_scaler),
                     ('onehot', OneHotEncoder(), ohe_cols)]
    
    # Create the ColumnTransformer object
    preprocessor = ColumnTransformer(transformers, remainder = 'passthrough')

    # Apply the transformations to the input DataFrame
    df_processed_array = preprocessor.fit_transform(dataframe)

    # Extract column names from transformers
    transformed_columns = []
    for name, transformer, features in transformers:
        if name in ['onehot', 'rb_scaler', 'mm_scaler']:
            transformed_columns += list(preprocessor.named_transformers_[name].get_feature_names_out(features))
        else:
            transformed_columns += features

    # Convert the NumPy array back to a DataFrame
    processed_data = pd.DataFrame(df_processed_array, columns = transformed_columns)

    return processed_data


def predict(homepath, model, input_data):
    """
    Make predictions using a trained model.
    input: [model] - model object
            [input_data] - data to make predictions
    return: [predictions] - predictions
    """
    threshold = pickle.load(open(homepath + '/models/threshold.pkl', 'rb'))
    probability = model.predict_proba(input_data)[:, 1]
    predictions = (probability >= threshold).astype(int)

    return predictions

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Make predictions using a CatBoostClassifier model')
    parser.add_argument('--homepath', required=True, help='Path to the project')
    parser.add_argument('--input_data', required=True, help='Path to the input data (CSV file)')
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Load the model
    model = load_model(args.homepath)

    # Load input data
    input_data = pd.read_csv(args.input_data)

    # Preprocess the input data
    # log transform
    log_cols = ['age', 'credit_score_age_ratio']
    # encoding
    ohe_cols = ['is_active_member', 'has_cr_card', 'geography', 'gender', 'balance_indicator', 'life_stage', 'cs_category', 'tenure_group']
    # re-scaling
    min_max_scaler = ['estimated_salary', 'balance', 'tenure', 'balance_salary_ratio', 'tenure_age_ratio',
    'estimated_salary_per_country', 'ltv', 'tenure_per_country', 'credit_score_per_gender']
    robust_scaler = ['credit_score', 'num_of_products', 'balance_per_age']

    preprocessed_data = preprocess_data(input_data, min_max_scaler, robust_scaler, ohe_cols, log_cols)

    # Make predictions
    predictions = predict(args.homepath, model, preprocessed_data)

    print('Predictions:', predictions)

if __name__ == "__main__":
    main()