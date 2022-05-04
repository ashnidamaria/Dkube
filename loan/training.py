#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import requests, os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import mlflow
from sklearn import metrics
from dkube.sdk import *
import joblib


# In[ ]:


inp_path = "/opt/dkube/in"
out_path = "/opt/dkube/out"


# In[ ]:


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", default=None, type=str, help="setup URL")
    parser.add_argument("--fs", dest="fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    dkubeURL = FLAGS.url
    fs = FLAGS.fs

    ########--- Read features from input FeatureSet ---########

    # Featureset API
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(URL=dkubeURL, token=authToken)

    # Read features
    feature_df = api.read_featureset(name = fs)  # output: data

    ########--- Train ---########
    
    X = feature_df.drop(["id","credit_default"], axis=1)
    y = feature_df["credit_default"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
    X_train.to_csv('/home/user3/workspace/LoanPA/loan/xtest.csv')
    print('restshbdjfsk')
    lr = RandomForestClassifier(random_state=100, class_weight = 'balanced')
    lr.fit(X_train,y_train)
    
    y_pred_train = lr.predict(X_train)    # Predict on train data.
    y_pred = lr.predict(X_test)   # Predict on test data.
    
    #######--- Calculating metrics ---############
    acc = metrics.accuracy_score(y_test, y_pred)

    print('Accuracy:', acc)  

    ########--- Logging metrics into Dkube via mlflow ---############
    mlflow.log_metric("Accuracy", acc)


    # Exporting model
    filename = os.path.join(out_path, "model.joblib")
    joblib.dump(lr, filename)

