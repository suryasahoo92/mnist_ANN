import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import joblib
import os
import logging

def prepare_data(df):
    """it is used to separate the dependent variables and independent features
    Args:
        df (pd.DataFrame): its the pandas DataFrame to
    Returns:
        tuple: it returns the tuples of dependent variables and independent variables
    """
    logging.info("Preparing the data by segregating the independent and dependent variables")

    (X_train_full,y_train_full),(X_test,y_test) = df.load_data()

    X_valid, X_train = X_train_full[:5000]/255., X_train_full[5000:]/255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    X_test = X_test/255.
    return X_train,y_train,X_valid, y_valid, X_test,y_test

def save_model(model, filename):
    """This saves the trained model to
    Args:
        model (python object): trained model to
        filename (str): path to save the trained model
    """
    logging.info("saving the trained model")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)  # ONLY CREATE IF MODEL_DIR DOES NOT EXISTS
    filepath = os.path.join(model_dir, filename)  # model/filename
    joblib.dump(model, filepath)
    logging.info(f"saved the trained model {filepath}")

def save_test(test_data,evaluated_data):
    model_dir = "test_result"
    os.makedirs(model_dir, exist_ok=True)
    test_data.to_csv(r'C:\Users\Admin\Desktop\DLCVNLP\ANN modular coding\test_result\result.csv', index=1)
    evaluated_data.to_csv(r'C:\Users\Admin\Desktop\DLCVNLP\ANN modular coding\test_result\result.csv', index=2)
