#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import argparse
import yaml
from sklearn import preprocessing as skpreprocessing

from dkube.sdk import *


# In[ ]:


inp_dir = "/opt/dkube/in"
out_path = "/opt/dkube/out"


# In[ ]:


if __name__ == "__main__":

    ########--- Parse for parameters ---########

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", dest="url", default=None, type=str, help="setup URL")
    parser.add_argument("--fs", dest="fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    fs = FLAGS.fs

    ########--- Get DKube client handle ---########

    dkubeURL = FLAGS.url
    # Dkube user access token for API authentication
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(URL=dkubeURL, token=authToken)

    ########--- Extract and load data  ---######
    
    loan = pd.read_csv(os.path.join(inp_dir, "train.csv"))

    ########--- Feature Engineering ---#######
    
    loan.columns = [("_".join(col.split(" "))).lower() for col in loan.columns]
    cat_cols = loan.select_dtypes("object").columns.to_list()
    num_cols = loan.select_dtypes(["float64", "int64"]).columns.to_list()
    loan.fillna(0, inplace = True)
    loan["years_in_current_job"] = loan["years_in_current_job"].apply(lambda x: str(x).strip("years"))
    # Create dummies
    loan_cat_df = pd.get_dummies(loan[cat_cols], drop_first=True)

    # # Add dummy columns
    loan = pd.concat([loan, loan_cat_df, ], axis=1)

    # Drop original columns
    loan.drop(cat_cols, axis=1, inplace=True)
    
    print ("Pre-processing completed")

    # Commit featureset
    resp = api.commit_featureset(name=fs, df=loan)
    print("featureset commit response:", resp)


# In[ ]:





# In[ ]:




