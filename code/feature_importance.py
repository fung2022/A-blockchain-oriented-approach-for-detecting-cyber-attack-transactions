# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 18:47:36 2022

@author: zhiqi
"""
# Let's load the packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from matplotlib import pyplot as plt

feature = ['gas_fee', 'value', 'action', 'spend_value', 'receive_value','internal_sum', 
                    'internal_median', 'internal_max', 'internal_count', 'internal_min', 
                    'internal_spend_sum', 'internal_spend_count', 'internal_receive_sum',
                    'internal_receive_count', 'token_address_nunique', 'token_address_count', 
                    'abnormal_index_sum', 'spend_token_address_nunique', 'spend_token_address_count',
                    'receive_token_address_nunique', 'receive_token_address_count',
                     'flash_token_address_nunique',
                    'flash_token_address_count', 'swap_token_address_nunique',
                    'swap_flashloan_sum', 'swap_index_sum',
                    'eth_spend_abnormal_index','eth_receive_abnormal_index',
                    'token_spend_abnormal_index','token_receive_abnormal_index','hash']

feature_name = ['gas-fee', 'ex-tx-volume', 'action-type', 'spend-volume', 'receive-volume','inter-tx-volume', 
                    'inter-ts-volume-50%', 'inter-ts-volume-max', 'inter-ts-count', 'inter-ts-volume-min', 
                    'inter-spend-volume', 'inter-spend-count', 'inter-receive-volume',
                    'inter-receive-count', 'to-ts-degree', 'to-ts-count', 
                    'to-ts-count-anomaly', 'to-ts-outdegree', 'spend-to-ts-count',
                    'to-ts-indegree', 'receive-to-ts-count',
                    'token-index', 'internal-index', 'contract-index',
                    'internal-token-index', 'flash-currency-count',
                    'flash-action-count', 'swap-currency-count',
                    'swap-flash-count', 'swap-action-count',
                    'spend-anomaly','receive-anomaly',
                    'to-spend-anomaly','to-receive-anomaly','hash']

def data_load(attack_type,feature_data_path,labeled_data_path):
    """
    load and split data into training dataset and test dataset
    para: attack_type
    para: feature_data_path
    para: labeled_data_path
    return: X_train, X_test, y_train, y_test
    """
    feature_data = pd.read_csv(feature_data_path,encoding='utf8',index_col=0)
    feature_data['compromised_address'] = [eval(x)[0] for x in feature_data.index]
    feature_data['target_block_number'] = [eval(x)[1] for x in feature_data.index]
    target_data = pd.read_csv(labeled_data_path,encoding='utf8')
    # attack_type = 'flash loan attack'
    temp_target_data = target_data[(target_data['type']==attack_type)].reset_index(drop=True) 
    temp_feature_data = feature_data[feature_data.index.isin(list(temp_target_data['index']))][feature].drop_duplicates()
    X = temp_feature_data.drop(['hash'],axis=1)
    Y = pd.DataFrame([1 if x in set(temp_target_data['hash']) else 0 for x in temp_feature_data['hash']])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    return X_train, X_test, y_train, y_test

plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.rcParams.update({'figure.figsize': (10.0, 8.0)})
plt.rcParams.update({'font.size': 12})

feature_data_path = 'data/feature_data_of_transactions.csv'
labeled_data_path = 'data/labeled_transaction_data.csv'
X_train, X_test, y_train, y_test = data_load('authority theft',feature_data_path,labeled_data_path)

###feature importance based on Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf.feature_importances_

sorted_idx = rf.feature_importances_.argsort()
plt.barh(np.array(feature_name)[sorted_idx], rf.feature_importances_[sorted_idx],color='royalblue')
plt.xlabel("Feature Importance")

###feature importance based on LGBM
train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_test, label=y_test)
params = {
    'learning_rate': 0.1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'max_depth': 5,
    'objective': 'binary',  # objective function
    'num_class': 1,
}
lgbm = lgb.train(params, train_data, valid_sets=[validation_data])

sorted_idx = lgbm.feature_importance().argsort()
plt.barh(np.array(feature_name)[sorted_idx], lgbm.feature_importance()[sorted_idx]/sum(lgbm.feature_importance()),color='royalblue')
plt.xlabel("Feature Importance")

