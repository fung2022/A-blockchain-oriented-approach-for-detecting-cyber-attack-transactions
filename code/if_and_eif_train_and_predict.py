# -*- coding: utf-8 -*-

import pandas as pd 
from sklearn import preprocessing
import time
from eif import iForest

def data_load(train_data_path,attack_data_path):
    """
    para train_data_path: feature data of histroical transaction
    para attack_data_path: labeled transactions of historical cyber-attacks
    para attack_type: type of cyber-attack
    return (feature_data,target_data)
    """
    feature_data = pd.read_csv(train_data_path,encoding='utf8',index_col=0).fillna(0)
    feature_data['compromised_address'] = [eval(x)[0] for x in feature_data.index]
    feature_data['target_block_number'] = [eval(x)[1] for x in feature_data.index]
    target_data = pd.read_csv(attack_data_path,encoding='utf8')
    
    return (feature_data,target_data)

def experiment_execution(feature_data,target_data,transaction_limit,contamination,sigma=False):
    """
    para feature_data: feature data of histroical transaction
    para target_data: labeled transactions of historical cyber-attacks
    para transaction_limit: The number of historical transactions feed into the model training and prediction
    para contamination: the rate of outlier in the whole data set
    return (predict_result,precision,time_cost)
    """
#    sigma = True
    start_time = time.time()
    target = target_data.to_dict(orient='records')
    predict_result = list()
    problem_list = []
    count = 1
    for x in target:
#        x = target[0]
        count = count + 1
        compromised_address = x['compromised_address']
        block_number = int(x['block_number'])
        current_data = feature_data[(feature_data['compromised_address']==compromised_address)&(feature_data['target_block_number']==block_number)].sort_values(by=['block_number'],ascending=False).head(transaction_limit)
        print("sample length: ",len(current_data))
        train_data = current_data[original_feature]
        
        if sigma ==True:
            select_columns = discrete_feature
            for y in train_data.columns:
                if y not in discrete_feature:
                    if abs(pd.Series(train_data[y]).skew())>1:
                        select_columns.append(y)
            train_data = train_data[select_columns]

        train_data.index = current_data['hash']
        if len(train_data)>=10:
            min_max_scaler = preprocessing.StandardScaler()
            train_data = min_max_scaler.fit_transform(train_data) 
            train_data = pd.DataFrame(train_data)
            train_data = train_data.to_numpy()
            IF = iForest(train_data,100,256 if len(current_data)*0.7>256 else int(len(current_data)*0.7),ExtensionLevel = 0) # ExtensionLevel=0 is the same as regular Isolation Forest
            IF_pred = list(IF.compute_paths(train_data))
            IF_threshold = sorted(IF_pred,reverse=True)[int(len(current_data)*0.05)]
            IF_pred = [1 if x >IF_threshold else 0 for x in IF_pred]
            
            EIF = iForest(train_data,100,256 if len(current_data)*0.7>256 else int(len(current_data)*0.7),ExtensionLevel = 4) #Extended isolation forest
            EIF_pred = list(EIF.compute_paths(train_data))
            EIF_threshold = sorted(EIF_pred,reverse=True)[int(len(current_data)*0.05)]
            EIF_pred = [1 if x >EIF_threshold else 0 for x in EIF_pred]            
            
            current_data["IF_pred"]=IF_pred
            current_data["EIF_pred"]=EIF_pred
            filter_result = current_data[
                                        (current_data['compromised_address']==compromised_address)& \
                                        (current_data['target_block_number']==block_number)& \
                                        (current_data['hash']==x['hash'])].head(1).to_dict(orient='records')
            filter_result[0]['type'] = x['type']
            predict_result = predict_result + filter_result
        else:
            problem_list.append(x)
    predict_result = pd.DataFrame(predict_result)[['hash','block_number','compromised_address',
                                                   'IF_pred','EIF_pred','type']]
    predict_result = predict_result.drop_duplicates()
    predict_result['label'] = [1 if x != 'normal transaction' else 0 for x in predict_result['type']] 
    end_time = time.time()
    time_cost = end_time-start_time
    print(predict_result.columns)
    return (predict_result,time_cost)

def metric(predict_result,attack_type):
    """
    para: predict_result
    para: attack_type
    return: result of metric
    """
    for x in ["IF_pred","EIF_pred"]:
        attack_result = predict_result[(predict_result['type']==attack_type)|(predict_result['type']=='normal transaction')].drop_duplicates()
        precision = sum(attack_result[attack_result['type']==attack_type][x])/sum(attack_result[x])
        recall = sum(attack_result[attack_result['type']==attack_type][x])/sum(attack_result[attack_result['type']==attack_type]['label'])
        f_score = 2*precision*recall/(precision+recall)
        print(x,"precision: ",precision,"recall: ",recall,"f_score: ",f_score)

#    return(precision,recall,f_score)

if __name__ == "__main__":
    original_feature = ['gas_fee', 'value', 'action', 'spend_value', 'receive_value','internal_sum', 
                        'internal_median', 'internal_max', 'internal_count', 'internal_min', 
                        'internal_spend_sum', 'internal_spend_count', 'internal_receive_sum',
                        'internal_receive_count', 'token_address_nunique', 'token_address_count', 
                        'abnormal_index_sum', 'spend_token_address_nunique', 'spend_token_address_count',
                        'receive_token_address_nunique', 'receive_token_address_count',
                        'token_index', 'internal_index', 'contract_index',
                        'internal_token_index', 'flash_token_address_nunique',
                        'flash_token_address_count', 'swap_token_address_nunique',
                        'swap_flashloan_sum', 'swap_index_sum',
                        'eth_spend_abnormal_index','eth_receive_abnormal_index',
                        'token_spend_abnormal_index','token_receive_abnormal_index']
    discrete_feature = ['action', 'token_index', 'internal_index', 'contract_index','internal_token_index']
    train_data_path = "data/feature_data_of_transactions.csv"
    attack_data_path = "data/labeled_transaction_data.csv"
    transaction_limit = 2000
    contamination = 0.05
    feature_data,target_data = data_load(train_data_path,attack_data_path)
    predict_result,time_cost = experiment_execution(feature_data,target_data,transaction_limit,contamination,sigma=False)
    metric(predict_result,'smart contract exploit')
    metric(predict_result,'flash loan attack')
    metric(predict_result,'authority theft')