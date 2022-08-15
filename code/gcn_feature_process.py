
import pandas as pd
from datetime import datetime 
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import threadpool
import queue
import time

# id = '0xfe9e8709d3215310075d67e3ed32a380ccf451c8'
def feature_process(id):
    """
    feature generation
    para address
    para data
    return final_feature
    """
    tx_data = temp_transaction_data[(temp_transaction_data['from']==id)|(temp_transaction_data['to']==id)].fillna(0)
    tx_data = tx_data[['block_number', 'hash', 'value', 'contract_address', 'gas_fee', 'day', 
                       'month', 'year', 'from', 'to']].drop_duplicates()
    final_feature = []
    """time feature"""
    valid_day_count = len(set(tx_data['day']))
    final_feature = final_feature + [valid_day_count]
    """token address feature"""
    token_address_set = set(tx_data['contract_address'])
    if 0 in token_address_set:
        token_address_num = len(token_address_set) - 1
    else:
        token_address_num = len(token_address_set)
    final_feature = final_feature + [token_address_num]
    """in and out degree"""
    in_degree = len(set(tx_data[tx_data['to']==id]['from']))
    out_degree = len(set(tx_data[tx_data['from']==id]['to']))
    final_feature = final_feature + [in_degree,out_degree]
    """tx_count feature"""
    tx_count_per_day = np.mean(tx_data['hash'].groupby([tx_data['day']]).nunique())
    tx_count_per_month = np.mean(tx_data['hash'].groupby([tx_data['month']]).nunique())
    tx_count_total = len(set(tx_data['hash']))
    final_feature = final_feature + [tx_count_per_day,tx_count_per_month,tx_count_total]

    eth_data = tx_data[tx_data['contract_address']==0]
    if len(eth_data)>0:
        """eth spend transaction feature"""
        eth_data_out = eth_data[eth_data['from']==id]
        if len(eth_data_out)>0:
            transaction_gas_fee = sum(eth_data_out['gas_fee']) #gas used
            avg_transaction_gas_fee = np.mean(eth_data_out['gas_fee']) # avg gas used
            if len(eth_data_out[eth_data_out['value']>0])>0:
                temp_data = eth_data_out[eth_data_out['value']>0]
                eth_tx_value_out = list(temp_data['value'].describe()) #spend value quantile
                eth_tx_count_out_per_day = np.mean(temp_data['hash'].groupby([temp_data['day']]).nunique())
                eth_tx_count_out_per_month = np.mean(temp_data['hash'].groupby([temp_data['month']]).nunique())
            else:
                eth_tx_value_out = [0,0,0,0,0,0,0,0]
                eth_tx_count_out_per_day = 0
                eth_tx_count_out_per_month = 0
        else:
            transaction_gas_fee = 0
            avg_transaction_gas_fee = 0
            eth_tx_value_out = [0,0,0,0,0,0,0,0]
            eth_tx_count_out_per_day = 0
            eth_tx_count_out_per_month = 0

        eth_data_in = eth_data[eth_data['to']==id]
        if len(eth_data_in)>0:
            """eth receive transaction feature"""
            eth_tx_value_in = list(eth_data_in['value'].describe()) #spend value quantile
            eth_tx_count_in_per_day = np.mean(eth_data_in['hash'].groupby([eth_data_in['day']]).nunique())
            eth_tx_count_in_per_month = np.mean(eth_data_in['hash'].groupby([eth_data_in['month']]).nunique())
        else:
            eth_tx_value_in = [0,0,0,0,0,0,0,0]
            eth_tx_count_in_per_day = 0
            eth_tx_count_in_per_month = 0
    else:
        transaction_gas_fee = 0
        avg_transaction_gas_fee = 0
        eth_tx_value_out = [0,0,0,0,0,0,0,0]
        eth_tx_count_out_per_day = 0
        eth_tx_count_out_per_month = 0
        eth_tx_value_in = [0,0,0,0,0,0,0,0]
        eth_tx_count_in_per_day = 0
        eth_tx_count_in_per_month = 0
    final_feature = final_feature + [transaction_gas_fee,avg_transaction_gas_fee]
    final_feature = final_feature + eth_tx_value_out
    final_feature = final_feature + [eth_tx_count_out_per_day,eth_tx_count_out_per_month]
    final_feature = final_feature + eth_tx_value_in
    final_feature = final_feature + [eth_tx_count_in_per_day,eth_tx_count_in_per_month]
    """erc spend transaction feature"""
    erc_data = tx_data[tx_data['contract_address']!=0]
    if len(erc_data)>0:
        erc_data_out = erc_data[erc_data['from']==id]
        if len(erc_data_out)>0:
            token_address_out_quantile = list(np.quantile(erc_data_out['hash'].groupby(erc_data_out['contract_address']).nunique(),[0.25,0.5,0.75]))
            token_address_out_avg = np.mean(erc_data_out['hash'].groupby(erc_data_out['contract_address']).nunique())
            erc_tx_count_out_per_day = np.mean(erc_data_out['hash'].groupby([erc_data_out['day']]).nunique())
            erc_tx_count_out_per_month = np.mean(erc_data_out['hash'].groupby([erc_data_out['month']]).nunique())
        else:
            token_address_out_quantile = [0,0,0]
            token_address_out_avg = 0
            erc_tx_count_out_per_day = 0
            erc_tx_count_out_per_month = 0
        
        """erc receive transaction feature"""
        erc_data_in = erc_data[erc_data['to']==id]
        if len(erc_data_in)>0:
            token_address_in_quantile = list(np.quantile(erc_data_in['hash'].groupby(erc_data_in['contract_address']).nunique(),[0.25,0.5,0.75]))
            token_address_in_avg = np.mean(erc_data_in['hash'].groupby(erc_data_in['contract_address']).nunique())

            erc_tx_count_in_per_day = np.mean(erc_data_in['hash'].groupby([erc_data_in['day']]).nunique())
            erc_tx_count_in_per_month = np.mean(erc_data_in['hash'].groupby([erc_data_in['month']]).nunique())
        else:
            token_address_in_quantile = [0,0,0]
            token_address_in_avg = 0
            erc_tx_count_in_per_day = 0
            erc_tx_count_in_per_month = 0
    else:
        token_address_out_quantile = [0,0,0]
        token_address_out_avg = 0
        erc_tx_count_out_per_day = 0
        erc_tx_count_out_per_month = 0
        token_address_in_quantile = [0,0,0]
        token_address_in_avg = 0
        erc_tx_count_in_per_day = 0
        erc_tx_count_in_per_month = 0
    final_feature = final_feature + token_address_out_quantile
    final_feature = final_feature + [token_address_out_avg,erc_tx_count_out_per_day,erc_tx_count_out_per_month]
    final_feature = final_feature + token_address_in_quantile
    final_feature = final_feature + [token_address_in_avg,erc_tx_count_in_per_day,erc_tx_count_in_per_month,id] 
    result_queue.put(final_feature)
    print(final_feature)

if __name__ == "__main__":

    eth_data = pd.read_csv('data/eth_data.csv',encoding='utf8',index_col=0)
    token_data = pd.read_csv('data/token_data.csv',encoding='utf8',index_col=0)
    token_data.columns = ['gas_fee', 'block_number', 'contract_address', 'from', 'to', 'block_date','value', 'hash']
    transaction_data = eth_data.append(token_data)
    target_data = pd.read_csv('data/labeled_transaction_data.csv',encoding='utf8')
    
    temp_target_data = target_data[target_data['type']=='authority theft']
    temp_transaction_data = transaction_data[transaction_data.index.isin(list(temp_target_data['index']))].fillna(0)
    temp_transaction_data['gas_fee'] = temp_transaction_data['gas_fee'].astype(float)
    temp_transaction_data['value'] = temp_transaction_data['value'].astype(float)
    temp_transaction_data['day'] = [str(x)[6:] for x in temp_transaction_data['block_date']]
    temp_transaction_data['month'] = [str(x)[4:6] for x in temp_transaction_data['block_date']]
    temp_transaction_data['year'] = [str(x)[:4] for x in temp_transaction_data['block_date']]
    
    address_list = list(set(list(temp_transaction_data['from']) + list(temp_transaction_data['to'])))
    print("start multi_threadpool")
    result_queue = queue.Queue()
    pool = threadpool.ThreadPool(64)
    predict_request = threadpool.makeRequests(feature_process, address_list)
    [pool.putRequest(req) for req in predict_request]
    pool.wait()
    
    feature_data = []
    while result_queue.qsize()>0:
        feature_data.append(result_queue.get())
        
    pd.DataFrame(feature_data).to_csv('data/authority_theft_attributes.csv',encoding='utf8')



# if __name__ == "__main__":
#     extract_feature_multi_process()



