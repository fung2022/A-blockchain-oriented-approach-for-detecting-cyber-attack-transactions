# -*- coding: utf-8 -*-

import pandas as pd 
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def xgb_train(X_train, X_test, y_train, y_test):
    """
    XGBoost model
    para: X_train
    para: X_test
    para: y_train
    para: y_test
    return: xgb_predict
    """
    param = {'max_depth':2, 'eta':1, 'objective':'binary:logistic' }
    num_round = 2
    xgb_model = xgb.train(param, xgb.DMatrix(X_train,y_train), num_round)
    # make prediction
    xgb_predict = xgb_model.predict(xgb.DMatrix(X_test))
    xgb_predict = [1 if x>=0.5 else 0 for x in xgb_predict]
    
    return xgb_predict

def random_forset_train(X_train, X_test, y_train, y_test):
    """
    random forest model
    para: X_train
    para: X_test
    para: y_train
    para: y_test
    return: rf_predict
    """
    rf_model = RandomForestClassifier() #using default parameter
    rf_model.fit(X_train,y_train)
    rf_predict = rf_model.predict(X_test)
    rf_predict = [1 if x>=0.5 else 0 for x in rf_predict]
    
    return rf_predict

def lgbm_train(X_train, X_test, y_train, y_test):
    """
    LightGBM model
    para: X_train
    para: X_test
    para: y_train
    para: y_test
    return: lgbm_pred
    """
    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_test, label=y_test)
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 4,
        'objective': 'binary',  # objective function
        'num_class': 1,
    }
    lgbm = lgb.train(params, train_data, valid_sets=[validation_data])
    lgbm_pred = lgbm.predict(X_test)
    lgbm_pred = [1 if x>=0.5 else 0 for x in list(lgbm_pred)]

    return lgbm_pred

def metric(y_true,y_predict):
    """
    compute the precision, recall and f1score
    para: y_true
    para: y_predict
    return: precision,recall,f1score
    """
    matrix = confusion_matrix(y_true, y_predict)
    precision = matrix[1][1] / (matrix[0][1] + matrix[1][1])
    recall = matrix[1][1] / (matrix[1][0] + matrix[1][1])
    f1score = 2*precision*recall/(precision+recall)
    print("precision:",precision," recall:",recall," f1score: ",f1score)
    return (precision,recall,f1score)

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
    temp_target_data = target_data[(target_data['type']==attack_type)].reset_index(drop=True) 
    temp_feature_data = feature_data[feature_data.index.isin(list(temp_target_data['index']))][original_feature].drop_duplicates()
    X = temp_feature_data.drop(['hash'],axis=1)
    Y = pd.DataFrame([1 if x in set(temp_target_data['hash']) else 0 for x in temp_feature_data['hash']])
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    original_feature = ['gas_fee', 'value', 'action', 'spend_value', 'receive_value','internal_sum', 
                        'internal_median', 'internal_max', 'internal_count', 'internal_min', 
                        'internal_spend_sum', 'internal_spend_count', 'internal_receive_sum',
                        'internal_receive_count', 'token_address_nunique', 'token_address_count', 
                        'abnormal_index_sum', 'spend_token_address_nunique', 'spend_token_address_count',
                        'receive_token_address_nunique', 'receive_token_address_count',
                        'token_index', 'internal_index', 'contract_index','token_spend_abnormal_index', 'token_receive_abnormal_index',
                        'internal_token_index', 'flash_token_address_nunique',
                        'flash_token_address_count', 'swap_token_address_nunique',
                        'swap_flashloan_sum', 'swap_index_sum','hash']
    
    feature_data_path = 'data/feature_data_of_transactions.csv'
    labeled_data_path = 'data/labeled_transaction_data.csv'
    
    X_train, X_test, y_train, y_test = data_load('flash loan attack',feature_data_path,labeled_data_path)
    
    #XGBoost
    xgb_predict = xgb_train(X_train, X_test, y_train, y_test)
    precision,recall,f1score = metric(list(y_test[0]),xgb_predict)
    
    #Random forest
    rf_predict = random_forset_train(X_train, X_test, y_train, y_test)
    precision,recall,f1score = metric(list(y_test[0]),rf_predict)
    
    #LightGBM
    lgbm_predict = lgbm_train(X_train, X_test, y_train, y_test)
    precision,recall,f1score = metric(list(y_test[0]),lgbm_predict)

