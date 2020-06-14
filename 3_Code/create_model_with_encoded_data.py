#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    get_ipython().run_line_magic('reload_ext', 'lab_black')
except ImportError as error:
    print(error)

import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.model_selection import KFold


# In[2]:


train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
submission = pd.read_csv("data/sample_submission.csv")

train_df.shape, test_df.shape, submission.shape


# In[3]:


def train_model(x_data, y_data, k=5):
    models = []

    k_fold = KFold(n_splits=k, shuffle=True, random_state=0)

    for train_idx, eval_idx in k_fold.split(x_data):
        X_train, y_train = x_data.iloc[train_idx], y_data[train_idx]
        X_eval, y_eval = x_data.iloc[eval_idx], y_data[eval_idx]

        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_eval, label=y_eval)

        wlist = [(dtrain, "train"), (dval, "eval")]

        params = {"objective": "reg:squarederror", "eval_metric": "mae", "seed": 0}

        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            verbose_eval=1000,
            evals=wlist,
        )
        models.append(model)

    return models


# In[4]:


import os

if not os.path.exists("encoded_result"):
    os.makedirs("encoded_result")


# In[5]:


for n in range(4, 10 + 1):
    encoded_train = pd.read_csv(f"encoded_data/encoded_rlt_train_{n}.csv", index_col=0)
    encoded_test = pd.read_csv(f"encoded_data/encoded_rlt_test_{n}.csv", index_col=0)

    X_train = encoded_train
    y_train = train_df.loc[:, "hhb":"na"]

    print(n)
    models = {}
    for label in y_train.columns:
        print("train column : ", label)
        models[label] = train_model(X_train, y_train[label])
        print()

    for col in models:
        preds = []
        for model in models[col]:
            preds.append(model.predict(xgb.DMatrix(encoded_test)))
        pred = np.mean(preds, axis=0)

        submission[col] = pred

    submission.to_csv(f"encoded_result/encoded_{n}.csv", index=False)

    for col in models:
        preds = []
        for model in models[col]:
            preds.append(model.predict(xgb.DMatrix(encoded_train)))
        pred = np.mean(preds, axis=0)

        train_df[col] = pred

    train_df.to_csv(f"encoded_result/train_encoded_{n}.csv", index=False)

