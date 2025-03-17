# train_global_model.py

import os
import logging
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
import joblib

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

if __name__=="__main__":
    csv_file = "global_labeled_data.csv"
    model_file = "global_model.pkl"

    if not os.path.exists(csv_file):
        # If no labeled data yet, create an empty file or just exit
        logging.info("No global_labeled_data.csv found. Creating empty one.")
        # create empty CSV with columns
        col_names = ["rMSSD","slope_rMSSD","HR","HF","slope_HF","alpha1","SynergyLabel"]
        df_empty = pd.DataFrame(columns=col_names)
        df_empty.to_csv(csv_file, index=False)
        logging.info("Empty global_labeled_data.csv created. No data to train on yet.")
        exit(0)

    # load the labeled data
    df = pd.read_csv(csv_file, names=None)
    # If it's empty, exit
    if len(df)==0:
        logging.info("global_labeled_data.csv is empty. Nothing to train.")
        exit(0)

    # assume columns: rMSSD, slope_rMSSD, HR, HF, slope_HF, alpha1, SynergyLabel
    if "SynergyLabel" not in df.columns:
        logging.error("No SynergyLabel column found in global_labeled_data.csv")
        exit(1)

    X = df.drop("SynergyLabel", axis=1).values
    y = df["SynergyLabel"].values.astype(int)

    if len(np.unique(y))<2:
        logging.warning("All synergy labels are the same. Model won't learn well.")
    
    logging.info(f"Training on {len(X)} labeled samples from {csv_file} ...")

    # We'll do a simple logistic regression (could be anything: SVC, RandomForest, etc.)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # Save model
    joblib.dump(model, model_file)
    logging.info(f"Trained model saved to {model_file}")
