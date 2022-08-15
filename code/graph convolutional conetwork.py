# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 16:55:47 2022

@author: EDZ
"""

import pandas as pd
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph
import networkx as nx
from tensorflow.keras import Model,metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from stellargraph.layer import GCNSupervisedGraphClassification
from sklearn import preprocessing
import sklearn

def read_graph(data):
    """
    load data
    """
    G=nx.Graph()
    for i in range(len(data)):
        G.add_edge(data['from'][i],data['to'][i])
    return G

"""正常交易与异常交易目标"""
eth_data = pd.read_csv('data/eth_data.csv',encoding='utf8',index_col=0)
token_data = pd.read_csv('data/token_data.csv',encoding='utf8',index_col=0)
transaction_data = eth_data[['from', 'to','hash','block_number']].append(token_data[['from', 'to','hash','block_number']])

target_data = pd.read_csv('data/labeled_transaction_data.csv',encoding='utf8')

"""authority theft"""
#def data_load()
temp_target_data = target_data[target_data['type']=='authority theft']

temp_transaction_data = pd.DataFrame()
for x in temp_target_data['index']:
    temp_transaction_data = temp_transaction_data.append(transaction_data[transaction_data.index==x].sort_values(by=['block_number'],ascending=False).head(10))

feature_data = pd.read_csv('data/authority_theft_attributes.csv',index_col=0,encoding='utf8').fillna(0)
feature_data.index = feature_data['41']
feature_data = feature_data.drop(['41'],axis=1)
standard_scaler = preprocessing.StandardScaler()
train_data = standard_scaler.fit_transform(feature_data) 
train_data = pd.DataFrame(train_data)
train_data['address'] = feature_data.index
train_data = train_data.reset_index(drop=True)

graphs = []
graph_labels = []
total_number = len(set(temp_transaction_data['hash']))
for x in set(temp_transaction_data['hash']):
    temp_graph_transaction = temp_transaction_data[temp_transaction_data['hash']==x].reset_index(drop=True)
    temp_graph = read_graph(temp_graph_transaction)
    for node_id, node_data in temp_graph.nodes(data=True):
        temp_feature = train_data[train_data['address']==node_id].reset_index(drop=True)
        node_data["feature"] = list(list(temp_feature.iloc[0,:-1])) 
    graph = StellarGraph.from_networkx(temp_graph, node_features="feature")
    graphs.append(graph)
    temp_label = 1 if x in set(temp_target_data['hash']) else 0
    graph_labels.append(temp_label)
    print("processing: ",len(graph_labels)/total_number)

summary = pd.DataFrame(
    [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
    columns=["nodes", "edges"],
)
summary.describe().round(1)

pd.Series(graph_labels).value_counts().to_frame()
graph_labels = pd.get_dummies(pd.Series(graph_labels), drop_first=True)

generator = PaddedGraphGenerator(graphs=graphs)

def create_graph_classification_model(generator):
    gc_model = GCNSupervisedGraphClassification(
        layer_sizes=[64, 64],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.1,
    )
    x_inp, x_out = gc_model.in_out_tensors()
    predictions = Dense(units=32, activation="relu")(x_out)
    predictions = Dense(units=16, activation="relu")(predictions)
    predictions = Dense(units=1, activation="sigmoid")(predictions)

    # Let's create the Keras model and prepare it for training
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=["acc"])

    return model

train_graphs=graph_labels.sample(frac=0.8)
test_graphs = graph_labels[~graph_labels.index.isin(train_graphs.index)]
generator = PaddedGraphGenerator(graphs=graphs)
model = create_graph_classification_model(generator)
epochs = 3  # maximum number of training epochs
keras_callbacks  = [EarlyStopping(monitor="acc", min_delta=0, patience=15, restore_best_weights=True),
                        ModelCheckpoint(filepath='gcn_weights.h5', monitor="val_auc",mode='auto',save_freq="epoch",save_best_only=True)]

gen = PaddedGraphGenerator(graphs=graphs)
train_gen = gen.flow(
    list(train_graphs.index - 1),
    targets=train_graphs.values,
    batch_size=32,
    symmetric_normalization=False,
)
test_gen = gen.flow(
    list(test_graphs.index - 1),
    targets=test_graphs.values,
    batch_size=1,
    symmetric_normalization=False,
)

history = model.fit(
    train_gen, epochs=epochs, validation_data=test_gen, verbose=1,shuffle=True,callbacks=keras_callbacks
)
model.compile(
    optimizer=Adam(lr=0.001,decay=1e-7), loss=binary_crossentropy, metrics=['acc',metrics.AUC(),metrics.Precision(),metrics.Recall()],
)
predict_proba = [x[0] for x in list(model.predict(test_gen))]
result_data= pd.DataFrame()
result_data['label'] = [x[0] for x in list(test_gen.targets)]
result_data['predict_proba'] = predict_proba
result_data['predict_label'] = [1 if x>=0.5 else 0 for x in predict_proba]
F_score = sklearn.metrics.f1_score(list(result_data['label']),list(result_data['predict_label']))
Precision = sklearn.metrics.precision_score(list(result_data['label']),list(result_data['predict_label']))
Recall = sklearn.metrics.recall_score(list(result_data['label']),list(result_data['predict_label']))
print(F_score,Precision,Recall)    