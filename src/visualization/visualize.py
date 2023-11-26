import pickle
import argparse
import pandas as pd
from typing import Union
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

def model_viz(homepath: str, input_data: Union[str, int, float], class_names=['Not Churn', 'Churn']):
    """
    Plot ROC AUC curve and confusion matrix for a single model.
    input: [homepath] - path to the project
              [input_data] - dataframe
              [class_names] - list of class names
    return: None
    """
    X = input_data.drop(columns=['exited', 'customer_id'])  # Drop the 'exited' and 'customer_id' columns from the input data
    y = input_data['exited']  # Get the 'exited' column as the target variable

    model = CatBoostClassifier()  # Create an instance of the CatBoostClassifier model
    model.load_model(homepath + 'models/model.cbm')  # Load the pre-trained model from the specified path
    threshold = pickle.load(open(homepath + 'models/threshold.pkl', 'rb'))  # Load the threshold value from the specified path
    y_prob = model.predict_proba(X)[: , 1]  # Get the predicted probabilities for the positive class
    y_pred = (y_prob >= threshold).astype(int)  # Convert the predicted probabilities to binary predictions based on the threshold

    fpr, tpr, _ = roc_curve(y, y_prob)  # Compute the false positive rate, true positive rate, and thresholds for the ROC curve
    roc_auc = auc(fpr, tpr)  # Compute the area under the ROC curve (AUC)

    cm = confusion_matrix(y, y_pred)  # Compute the confusion matrix

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create a figure with two subplots

    # Plot Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title(f'{type(model).__name__}\nThreshold = {threshold:.2f}\nAUC = {roc_auc:.2f}')  # Set the title of the first subplot
    axes[0].set_xlabel('Predicted')  # Set the x-axis label of the first subplot
    axes[0].set_ylabel('True')  # Set the y-axis label of the first subplot

    # Plot ROC AUC curve
    axes[1].plot(fpr, tpr, color='darkorange', lw=2)  # Plot the ROC curve
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Plot the diagonal line
    axes[1].set_xlim([0.0, 1.0])  # Set the x-axis limits of the second subplot
    axes[1].set_ylim([0.0, 1.05])  # Set the y-axis limits of the second subplot
    axes[1].set_xlabel('False Positive Rate')  # Set the x-axis label of the second subplot
    axes[1].set_ylabel('True Positive Rate')  # Set the y-axis label of the second subplot
    axes[1].set_title('ROC Curve')  # Set the title of the second subplot

    plt.show()  # Display the figure

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Plot ROC AUC curve and confusion matrix for a single model')
    parser.add_argument('--homepath', required=True, help='Path to the project')
    parser.add_argument('--input_data', required=True, help='Path to the input data (CSV or other supported format)')

    # Parse command-line arguments
    args = parser.parse_args()

    # Load input data
    df = pd.read_csv(args.input_data)  # Load the input data from the specified path (assuming it is in CSV format)

    # Call the model_viz function
    model_viz(args.homepath, df)

if __name__ == "__main__":
    main()