import pickle
import inflection
import numpy as np
import pandas as pd
from typing import Union
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, RobustScaler, OneHotEncoder, FunctionTransformer

class TopBankChurnPrediction(object):
    def __init__(self):
        self.homepath = 'client_churn_prediction/src/model/'
        #self.homepath = 'model/'
        self.threshold = pickle.load(open(self.homepath + '/threshold.pkl', 'rb'))



    def loading_data(self, df: Union[int, float, str]):
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
    

    def build_features(self, df: Union[int, float, str]):
        """
        Build features for the model.
        input: [df] - dataframe
        return: [df] - dataframes with modifications
        """

        df['balance_salary_ratio'] = df['balance']/df['estimated_salary']
        # credit_score_age_ratio
        df['credit_score_age_ratio'] = df['credit_score']/df['age']
        # tenure_age_ratio
        df['tenure_age_ratio'] = df['tenure']/df['age']
        # life_stage
        df['life_stage'] = df['age'].apply(lambda x: 'adolescence' if x <= 20 else 'adulthood' if (
            x > 20) & (x <= 35) else 'middle_age' if (x > 35) & (x <= 50) else 'senior')
        # balance_age
        balance_age = df.loc[:, ['age', 'balance']].groupby('age').mean().reset_index()
        balance_age.columns = ['age', 'balance_per_age']
        df = pd.merge(df, balance_age, on = 'age', how = 'left')
        # LTV
        balance_tenure = df.loc[:, ['tenure', 'balance']].groupby('tenure').mean().reset_index()
        balance_tenure.columns = ['tenure', 'ltv']
        df= pd.merge(df, balance_tenure, on = 'tenure', how = 'left')

        return df
    

    def preprocess_data(self, df: Union[int, float, str]):
        """
        This function takes a Pandas DataFrame as input, along with three lists of column names (`min_max_scaler`, `robust_scaler`, `standard_scaler` and `cols_ohe`). The function applies a series of transformations to the input DataFrame, including one-hot encoding, label encoding, min-max scaling, standard scaling and robust scaling.

        Args:
            dataframe (Union[int, float, str]): Dataframe with all features

        Returns:
            dataframe: dataframe with transformations
        """

        transformers = []

        # log transform
        log_cols = ['age', 'credit_score_age_ratio']

        # encoding
        ohe_cols = ['is_active_member', 'has_cr_card', 'geography', 'gender', 'balance_indicator', 'life_stage', 'cs_category', 'tenure_group']

        # re-scaling
        min_max_scaler = ['estimated_salary', 'balance', 'tenure', 'balance_salary_ratio', 'tenure_age_ratio',
                            'estimated_salary_per_country', 'ltv', 'tenure_per_country', 'credit_score_per_gender']

        robust_scaler = ['credit_score', 'num_of_products', 'balance_per_age']

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
        df_processed_array = preprocessor.fit_transform(df)

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


    def predict(self, model, input_data):
        """
        Make predictions using a trained model.
        input: [model] - model object
                [input_data] - data to make predictions
        return: [predictions] - predictions
        """
        threshold = self.threshold
        probability = model.predict_proba(input_data)[:, 1]
        predictions = (probability >= threshold).astype(int)
    
        return predictions