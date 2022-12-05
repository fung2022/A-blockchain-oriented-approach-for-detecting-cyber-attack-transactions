# -*- coding: utf-8 -*-

import pandas as pd
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn import preprocessing
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.vae import VAE
from pyod.models.feature_bagging import FeatureBagging

def metric(predict_result,attack_type):
    """
    para: predict_result
    para: attack_type
    return: result of metric
    """
    attack_result = predict_result[(predict_result['type']==attack_type)|(predict_result['type']=='normal transaction')].drop_duplicates()
    metric_result = []
    for x in ['Cluster-based Local Outlier Factor (CBLOF)',
               'Histogram-base Outlier Detection (HBOS)', 'K Nearest Neighbors (KNN)',
               'Average KNN', 'Local Outlier Factor (LOF)', 'One-class SVM (OCSVM)',
               'Feature Bagging (FB)',
               'Deep One-Class Classifier with AutoEncoder (DeepSVDD)',
               'Variational Auto Encoder (VAE)']:
        precision = sum(attack_result[attack_result['type']==attack_type][x])/sum(attack_result[x])
        recall = sum(attack_result[attack_result['type']==attack_type][x])/sum(attack_result[attack_result['type']==attack_type]['label'])
        f_score = 2*precision*recall/(precision+recall)
        print(x,"precision: ",precision,"recall: ",recall,"f_score: ",f_score)
        metric_result.append([x,precision,recall,f_score])
    
    result = pd.DataFrame(metric_result)
    result.columns = ['Algorithm','precision','recall','f_score']
    return(result)

def data_load():
    """
    load dataset
    """
    final_data = pd.read_csv('data/feature_data_of_transactions.csv',encoding='utf8',index_col=0)
    final_data['compromised_address'] = [eval(x)[0] for x in final_data.index]
    final_data['target_block_number'] = [eval(x)[1] for x in final_data.index]
    train_data = pd.read_csv('data/train_data.csv',encoding='utf8',index_col=0).to_dict(orient='records')
    test_data = pd.read_csv('data/test_data.csv',encoding='utf8',index_col=0).to_dict(orient='records')
    return(final_data,train_data,test_data)

"""model setting"""
outliers_fraction = 0.05
classifiers = {
    'Angle-based Outlier Detector (ABOD)':
        ABOD(contamination=outliers_fraction),
    'Cluster-based Local Outlier Factor (CBLOF)':
        CBLOF(contamination=outliers_fraction,
              check_estimator=False, random_state=0),
    'Histogram-base Outlier Detection (HBOS)': HBOS(
        contamination=outliers_fraction),
    'Feature Bagging (FB)': FeatureBagging(check_estimator=False),
    'K Nearest Neighbors (KNN)': KNN(
        contamination=outliers_fraction),
    'Average KNN': KNN(method='mean',
                       contamination=outliers_fraction),
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=35, contamination=outliers_fraction),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'Deep One-Class Classifier with AutoEncoder (DeepSVDD)': DeepSVDD(
        use_ae=False , epochs=5, contamination=outliers_fraction,
                   random_state=10),
    'Variational Auto Encoder (VAE)': VAE(
            epochs=10, encoder_neurons=[30, 64, 128],decoder_neurons=[30, 64, 128],contamination=outliers_fraction, gamma=0.8, capacity=0.2)
}
"""feature scope"""
original_feature = ['gas_fee', 'value', 'action', 'spend_value', 'receive_value','internal_sum', 
                    'internal_median', 'internal_max', 'internal_count', 'internal_min', 
                    'internal_spend_sum', 'internal_spend_count', 'internal_receive_sum',
                    'internal_receive_count', 'token_address_nunique', 'token_address_count', 
                    'abnormal_index_sum', 'spend_token_address_nunique', 'spend_token_address_count',
                    'receive_token_address_nunique', 'receive_token_address_count',
                    'token_index', 'internal_index', 'contract_index',
                    'internal_token_index', 'flash_token_address_nunique',
                    'flash_token_address_count', 'swap_token_address_nunique',
                    'swap_flashloan_sum', 'swap_index_sum']

"""start to trian and predict"""
def train_and_predict(final_data,target_data):
    predict_result = pd.DataFrame()
    problem_list = []
    for x in target_data:
        print(x)
        compromised_address = x['compromised_address']
        block_number = int(x['block_number'])
        current_data = final_data[(final_data['compromised_address']==compromised_address)&(final_data['target_block_number']==block_number)]
        print("sample length: ",len(current_data))
        if len(current_data)>=10:
            feature_data = current_data[original_feature]
            feature_data.index = current_data['hash']
        
            min_max_scaler = preprocessing.StandardScaler()
            train_data = min_max_scaler.fit_transform(feature_data) 
            """adopting different to train and predict"""
            for i, (clf_name, clf) in enumerate(classifiers.items()):
                print("clf_name:", clf_name)
                clf.fit(train_data)
                y_pred = clf.predict(train_data)
                current_data[clf_name]=y_pred
            filter_result = current_data[
                                        (current_data['compromised_address']==compromised_address)& \
                                        (current_data['target_block_number']==block_number)& \
                                        (current_data['hash']==x['hash'])]
            filter_result['type'] = x['type'] 
            filter_columns = ['hash','block_number','compromised_address',
                               'Cluster-based Local Outlier Factor (CBLOF)',
                               'Histogram-base Outlier Detection (HBOS)', 
                               'K Nearest Neighbors (KNN)', 
                               'Average KNN',
                               'Local Outlier Factor (LOF)',
                               'One-class SVM (OCSVM)',
                               'Feature Bagging (FB)',
                               'Deep One-Class Classifier with AutoEncoder (DeepSVDD)',
                               'Variational Auto Encoder (VAE)','type']
            predict_result = predict_result.append(filter_result[filter_columns])
        else:
            problem_list.append([x,len(current_data)])
    predict_result['label'] = [1 if x!='normal transaction' else 0 for x in predict_result['type']]
    return(predict_result)

final_data,train_data,test_data = data_load()
train_predict_result = train_and_predict(final_data,train_data)
test_predict_result = train_and_predict(final_data,test_data)
#pd.DataFrame(predict_result).to_csv('data/comparative_results_of_uml.csv')
"""evaluation"""
smart_contract_exploit_result = metric(test_predict_result,'smart contract exploit')
flash_loan_result = metric(test_predict_result,'flash loan attack')
authority_theft_result = metric(test_predict_result,'authority theft')


