import argparse
import pandas as pd
from catboost import CatBoostClassifier
import pickle

def train_model(homepath: str, train_data_path: str, save_path: str):
    """
    Train a CatBoostClassifier on the provided training data and save the model.

    Args:
        homepath (str): Path to the project.
        train_data_path (str): Path to the training data (CSV or other supported format).
        save_path (str): Path to save the trained model.

    Returns:
        None
    """
    # Load training data
    train_data = pd.read_csv(train_data_path)

    # Separate features and target variable
    X = train_data.drop(columns=['exited', 'customer_id'])
    y = train_data['exited']

    # Initialize and train the model
    seed = 42
    model = CatBoostClassifier(learning_rate = 0.02198927031636029, depth = 7, n_estimators = 1500, scale_pos_weight = proportion, random_state = seed, verbose = False) # 0.43 F1
    model.fit(X, y)

    # Save the trained model
    model.save_model(f"{homepath}/models/model.cbm")

    print("Model trained and saved successfully.")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Train a CatBoostClassifier and save the model')
    parser.add_argument('--homepath', required=True, help='Path to the project')
    parser.add_argument('--train_data', required=True, help='Path to the training data (CSV or other supported format)')
    parser.add_argument('--save_path', required=True, help='Path to save the trained model')

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the train_model function
    train_model(args.homepath, args.train_data, args.save_path)

if __name__ == "__main__":
    main()