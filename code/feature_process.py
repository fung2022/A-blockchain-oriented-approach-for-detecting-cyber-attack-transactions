# -*- coding: utf-8 -*-

import pandas as pd
import datetime

def eth_data_process(eth_data,address):
    """
    feature generation based on external transaction
    para: eth_data
    papa: address
    return: eth_data
    """
    eth_data = pd.DataFrame(eth_data)
    eth_data['value'] = eth_data['value'].astype(float)
    eth_data['fee'] = eth_data['fee'].astype(float)
    eth_data['value'] = [float(x)/10**18 for x in eth_data['value']]
    eth_data['fee'] = [float(x)/10**18 for x in eth_data['fee']]
    eth_data['contract_transaction'] = [1 if x==0 else 0 for x in eth_data['value']]
    action=[]
    for x in range(len(eth_data)):
        if eth_data['from'][x]==address:
            action.append(-1)
        elif eth_data['to'][x]==address:
            action.append(1)
        else:
            action.append(0)
    eth_data['action'] = action
    return(eth_data)

def internal_data_process(internal_data,address):
    """
    feature generation based on internal transaction
    para: internal_data
    return: internal_data
    """
    internal_data = pd.DataFrame(internal_data)
    internal_data['value'] = internal_data['value'].astype(float)
    internal_data['value'] = [x/10**18 for x in internal_data['value']]
    internal_data = internal_data[internal_data['value']>0].reset_index(drop=True)
    action=[]
    for x in range(len(internal_data)):
        if internal_data['from'][x]==address:
            action.append(-1)
        elif internal_data['to'][x]==address:
            action.append(1)
        else:
            action.append(0)
    internal_data['action'] = action
    value_data = pd.DataFrame(internal_data.groupby(['hash'])['value'].agg({'count','sum','max','min','median'}))
    value_data = value_data[['count','sum','max','min','median']]
    spend_data = internal_data[(internal_data['action']==-1)].groupby(['hash'])['value'].agg({'count','sum'})
    spend_data = spend_data[['count','sum']]
    receive_data = internal_data[(internal_data['action']==1)].groupby(['hash'])['value'].agg({'count','sum'})
    receive_data = receive_data[['count','sum']]
    internal_data_result = pd.merge(value_data,spend_data,how='left',right_index=True,left_index=True)
    internal_data_result = pd.merge(internal_data_result,receive_data,how='left',right_index=True,left_index=True)
    internal_data_result.columns = ['internal_count', 'internal_sum', 'internal_max', 'internal_min', 'internal_median',
                                'internal_spend_count', 'internal_spend_sum', 'internal_receive_count', 'internal_receive_sum']
    return(internal_data_result.fillna(0))

def token_data_process(token_data,address):
    """
    feature generation based on external transaction
    para: token_data
    para: address
    return: token_data
    """
    token_data = pd.DataFrame(token_data)
#    token_data = temp_token_data.copy()
#    address = target_address
    action=[]
    for x in range(len(token_data)):
        if token_data['from'][x]==address:
            action.append(-1)
        elif token_data['to'][x]==address:
            action.append(1)
        else:
            action.append(0)
    token_data['action'] = action
    token_data['value'] = token_data['value'].astype(float)
    
    """statistics of crypto currencies"""
    temp_data = token_data[(token_data['from']==address)|(token_data['to']==address)]
    currency_statistics = pd.DataFrame(temp_data['value'].groupby([temp_data['token_address']]).describe()).fillna(0)
    currency_statistics['token_address'] = currency_statistics.index
    
    """apply three sigma process to process token transaction"""
    abnormal_index = []
    for x in range(len(token_data)):
        try:
            target = currency_statistics[currency_statistics['token_address']==token_data['token_address'][x]].to_dict(orient='record')
            if token_data['value'][x]>target[0]['mean']+1.5*target[0]['std'] and target[0]['count']>10:
                abnormal_index.append(1)
            else:
                abnormal_index.append(0)
        except:
            abnormal_index.append(0)
    token_data['abnormal_index'] = abnormal_index
    abnormal_index_data = token_data[token_data['abnormal_index']==1]
    abnormal_index_data['spend_abnormal_index'] = [1 if x==-1 else 0 for x in abnormal_index_data['action']]
    abnormal_index_data['receive_abnormal_index'] = [1 if x==1 else 0 for x in abnormal_index_data['action']]
    abnormal_index_data = abnormal_index_data.groupby(['hash'])['spend_abnormal_index','receive_abnormal_index'].agg({'sum'})
    """summary statistics of outgoing and incoming transactions"""
    summary_data = token_data.groupby(['hash'])['token_address','abnormal_index'].agg({'nunique','count','sum'})
    summary_data = summary_data[[( 'token_address', 'nunique'),
                                ( 'token_address',   'count'),
                                ('abnormal_index',     'sum')]]
    """statistics of outgoing transaction"""
    spend_data = token_data[(token_data['action']==1)].groupby(['hash'])['token_address','value'].agg({'nunique','count','sum'})
    spend_data = spend_data[[('token_address', 'nunique'),
                            ('token_address',   'count'),
                            (        'value',     'sum')]]
    """statistics of incoming transaction"""
    receive_data = token_data[(token_data['action']==-1)].groupby(['hash'])['token_address','value'].agg({'nunique','count','sum'})
    receive_data = receive_data[[('token_address', 'nunique'),
                                ('token_address',   'count'),
                                (        'value',     'sum')]]
    """merge"""
    token_data_result = pd.merge(summary_data,spend_data,how='left',right_index=True,left_index=True)
    token_data_result = pd.merge(token_data_result,receive_data,how='left',right_index=True,left_index=True)  
    token_data_result = pd.merge(token_data_result,abnormal_index_data,how='left',right_index=True,left_index=True).fillna(0)
    return(token_data_result)
    
def flashloan_data_process(flashloan_data):
    """
    feature generation based on calls of flash loan function
    para: flashloan_data
    return: flashloan_data
    """
    flashloan_data = pd.DataFrame(flashloan_data)
    flashloan_data = flashloan_data.groupby(['hash'])['token_address'].agg({'nunique','count'})
    flashloan_data.columns = ['flash_token_address_nunique','flash_token_address_count']
    return(flashloan_data)
    
def swap_data_process(swap_data):
    """
    feature generation based on calls of swap function
    para: swap_data
    return: swap_data
    """
    swap_data = pd.DataFrame(swap_data)
    swap_data['amount0_out'] = swap_data['amount0_out'].astype(float)
    swap_data['amount0_in'] = swap_data['amount0_in'].astype(float)
    swap_data['amount1_out'] = swap_data['amount1_out'].astype(float)
    swap_data['amount1_in'] = swap_data['amount1_in'].astype(float)
    
    swap_data['swap_flashloan'] = [1 if (swap_data['amount0_out'][x]>0 and swap_data['amount0_in'][x]>0) or (swap_data['amount1_out'][x]>0 and swap_data['amount1_in'][x]>0) else 0 for x in range(len(swap_data))]
    swap_data['swap_index'] = [1 for x in range(len(swap_data))]
    swap_data = swap_data[['hash', 'token_address', 'swap_flashloan', 'swap_index']]
     
    swap_data = swap_data.groupby(['hash'])['token_address','swap_flashloan','swap_index'].agg({'nunique','sum'})
    swap_data = swap_data[[( 'token_address', 'nunique'),
                                    ('swap_flashloan', 'nunique'),
                                    (    'swap_index', 'nunique')]]
    swap_data.columns = ['swap_token_address_nunique','swap_flashloan_sum','swap_index_sum']
    return(swap_data)

def timestamp2string(timeStamp):
    try:
        d = datetime.datetime.fromtimestamp(timeStamp)
        str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
        # 2015-08-28 16:43:37.283000'
        return str1
    except Exception as e:
        print(e)
        return ''

def main(address,accident_block_number,eth_data,internal_data,token_data,flashloan_data,swap_data):
#    address = "0x40b9b889a21ff1534d018d71dc406122ebcf3f5a"
#    accident_block_number = 1506290
#    eth_data,internal_data,token_data,flashloan_data,swap_data = main_data_process(address,accident_block_number)
    
    if len(eth_data)>0:
        eth_data_result = eth_data_process(eth_data,address)
#        eth_data_result['block_time'] = [timestamp2string(x) for x in eth_data_result['block_time']]
#        eth_data_result['block_time'] = [x[:13] for x in eth_data_result['block_time']]
    else:
        eth_data_result = pd.DataFrame()
    if len(internal_data)>0:
        internal_data_result = internal_data_process(internal_data,address)
#        eth_data_result['block_time'] = [x[:13] for x in eth_data_result['block_time']]
    else:
        internal_data_result = pd.DataFrame()
    if len(token_data)>0:
        token_data_result = token_data_process(token_data,address)
        token_data_result.columns = ['token_address_nunique','token_address_count','abnormal_index_sum',
                                    'spend_token_address_nunique','spend_token_address_count','spend_value_sum',
                                    'receive_token_address_nunique','receive_token_address_count','receive_value_sum',
                                    'token_spend_abnormal_index','token_receive_abnormal_index']
    else:
        token_data_result = pd.DataFrame()
    
    """计算eth本币的支出和收入是否产生异常"""
    eth_data_summary = pd.merge(eth_data_result,internal_data_result,how='left',right_index=True,left_on=['hash']).fillna(0)
    spend_value = []
    receive_value = []
    if len(internal_data)>0:
        for x in range(len(eth_data_summary)):
            if eth_data_summary['action'][x]==-1:
                temp_value = eth_data_summary['value'][x] + eth_data_summary['internal_spend_sum'][x]
                spend_value.append(temp_value)
            else:
                spend_value.append(eth_data_summary['internal_spend_sum'][x])
            
            if eth_data_summary['action'][x]==1:
                temp_value = eth_data_summary['value'][x] + eth_data_summary['internal_receive_sum'][x]
                receive_value.append(temp_value)
            else:
                receive_value.append(eth_data_summary['internal_receive_sum'][x])   
    else:
        for x in range(len(eth_data_summary)):
            if eth_data_summary['action'][x]==-1:
                temp_value = eth_data_summary['value'][x]
                spend_value.append(temp_value) 
            else:
                spend_value.append(0)
            if eth_data_summary['action'][x]==1:
                temp_value = eth_data_summary['value'][x]
                receive_value.append(temp_value)
            else:
                receive_value.append(0)
    """three sigma process of outgoing and incoming value"""
    eth_data_summary['spend_value'] = spend_value
    eth_data_summary['receive_value'] = receive_value
    spend_statistics = eth_data_summary[eth_data_summary['spend_value']!=0]['spend_value'].describe().fillna(0)
    receive_statistics = eth_data_summary[eth_data_summary['receive_value']!=0]['receive_value'].describe().fillna(0)
    eth_spend_abnormal_index = [1 if x>int(spend_statistics['mean']+1.5*spend_statistics['std']) else 0 for x in spend_value]
    eth_receive_abnormal_index = [1 if x>int(receive_statistics['mean']+1.5*receive_statistics['std']) else 0 for x in receive_value]
    eth_data_summary['eth_spend_abnormal_index'] = eth_spend_abnormal_index
    eth_data_summary['eth_receive_abnormal_index'] = eth_receive_abnormal_index
    
    
    if len(internal_data)==0:
        eth_data_summary['internal_sum'], eth_data_summary['internal_median'], eth_data_summary['internal_max'], \
        eth_data_summary['internal_count'], eth_data_summary['internal_min'], eth_data_summary['internal_spend_sum'], \
        eth_data_summary['internal_spend_count'], eth_data_summary['internal_receive_sum'], eth_data_summary['internal_receive_count'] = \
        [0,0,0,0,0,0,0,0,0]

    """make sure the type of transaction"""
    final_data = pd.merge(eth_data_summary,token_data_result,how='left',left_on=['hash'],right_index=True).fillna(0)
    if len(token_data)==0:
        final_data['token_address_nunique'], final_data['token_address_count'], final_data['abnormal_index_sum'], \
        final_data['spend_token_address_nunique'], final_data['spend_token_address_count'], final_data['spend_value_sum'], \
        final_data['receive_token_address_nunique'], final_data['receive_token_address_count'], final_data['receive_value_sum'], \
        final_data['token_spend_abnormal_index'], final_data['token_receive_abnormal_index'] = [0,0,0,0,0,0,0,0,0,0,0]
    
    final_data = final_data.reset_index(drop=True)
    token_index = [1 if final_data['value'][x]==0 and final_data['internal_count'][x]==0 and final_data['token_address_count'][x]>0 else 0 for x in range(len(final_data))]
    internal_index =  [1 if final_data['internal_count'][x]>0 and final_data['token_address_count'][x]==0 else 0 for x in range(len(final_data))]
    contract_index= [1 if final_data['value'][x]==0 and final_data['internal_count'][x]==0 and final_data['token_address_count'][x]==0 else 0 for x in range(len(final_data))]
    internal_token_index = [1 if final_data['internal_count'][x]>0 and final_data['token_address_count'][x]>0 else 0 for x in range(len(final_data))]
    
    final_data["token_index"] = token_index
    final_data["internal_index"] = internal_index
    final_data["contract_index"] = contract_index
    final_data["internal_token_index"] = internal_token_index
    final_data = final_data.sort_values(by=['block_number'],ascending=True)
    
    """temperol feature based on calls of specific functions"""
    if len(flashloan_data)>0:
        flashloan_data = flashloan_data_process(flashloan_data)
        final_data = pd.merge(final_data,flashloan_data,how='left',left_on=['hash'],right_index=True).fillna(0)
    else:
        final_data['flash_token_address_nunique'] = [0 for x in final_data['hash']]
        final_data['flash_token_address_count'] = [0 for x in final_data['hash']]
    
    if len(swap_data)>0:
        swap_data = swap_data_process(swap_data)
        final_data = pd.merge(final_data,swap_data,how='left',left_on=['hash'],right_index=True).fillna(0)
    else:
        final_data['swap_token_address_nunique'] = [0 for x in final_data['hash']]
        final_data['swap_flashloan_sum'] = [0 for x in final_data['hash']]
        final_data['swap_index_sum'] = [0 for x in final_data['hash']]
    return(final_data,swap_data)

if __name__ == "__main__":
    eth_data = pd.read_csv('data/eth_data.csv',encoding='utf8',index_col=0)
    internal_data = pd.read_csv('data/internal_data.csv',encoding='utf8',index_col=0)
    token_data = pd.read_csv('data/token_data.csv',encoding='utf8',index_col=0)
    flashloan_data = pd.read_csv('data/flashloan_data.csv',encoding='utf8',index_col=0)
    swap_data = pd.read_csv('data/swap_data.csv',encoding='utf8',index_col=0)
    target_data =[eval(x) for x in list(set(list(eth_data.index)+list(internal_data.index)+list(token_data.index)+list(flashloan_data.index)+list(swap_data.index)))]
    problem_list = []
    final_feature = pd.DataFrame()
    for x in target_data:
        target_address = x[0]
        target_blocknumber = x[1]
        print(x,target_data.index(x))
        try:
            if str((target_address,target_blocknumber)) in eth_data.index:
                temp_eth_data = eth_data[eth_data.index==str((target_address,target_blocknumber))]
            else:
                temp_eth_data = pd.DataFrame()
            if str((target_address,target_blocknumber)) in internal_data.index:
                temp_internal_data = internal_data[internal_data.index==str((target_address,target_blocknumber))]
            else:
                temp_internal_data = pd.DataFrame()
            if str((target_address,target_blocknumber)) in token_data.index:
                temp_token_data = token_data[token_data.index==str((target_address,target_blocknumber))]
            else:
                temp_token_data = pd.DataFrame()
            if str((target_address,target_blocknumber)) in flashloan_data.index:
                temp_flashloan_data = flashloan_data[flashloan_data.index==str((target_address,target_blocknumber))]
            else:
                temp_flashloan_data = pd.DataFrame()
            if str((target_address,target_blocknumber)) in swap_data.index:
                temp_swap_data = swap_data[swap_data.index==str((target_address,target_blocknumber))]
            else:
                temp_swap_data = pd.DataFrame()
            temp_feature,b = main(target_address,target_blocknumber,temp_eth_data,temp_internal_data,temp_token_data,temp_flashloan_data,temp_swap_data)
            temp_feature.index = [(target_address,target_blocknumber) for x in range(len(temp_feature))]
            final_feature = final_feature.append(temp_feature)
        except Exception as e:
            print(str(e))
    
