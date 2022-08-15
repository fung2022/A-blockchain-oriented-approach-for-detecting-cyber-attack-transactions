# A blockchain-oriented approach for detecting cyber-attack transactions 


We have multiple labeled cyber-attack transactions and multiple programs. Please read the program introduction carefully before executing.

Four types of transaction, such as normal transaction and three types of cyber-attack transaction on different account-based blockchains, are selected for the application and validation of our proposed method. The dataset contains three types of data including the original transaction information, labeled cyber-attack transactions and feature data. the detail information are shown as follow:

* The original transaction information: eth_data.csv, internal_data.csv, token_data.csv, swap_data.csv and floash_loan_data.csv.
* The labeled cyber-attack transactions: labled_transaction_data.csv.
* Feature data: feature_data_of_transactions.csv.

All of these data are available for download throught network disk. URL: https://pan.baidu.com/s/1ZxjtUaYfGOGJEvzy56wS3w password: xd5w

The codes of comparative experiment are stored in code folder. The file named feature_process.py is the execution file corresponding to procedure of feature generation and three sigma process. The weighted_extended_isolation_forest.py is the main code of our proposed algorithm mentioned in section 3. The files of unsupervised machine learning methods applied in section 4 are weif_train_and_predict.py (The execution code of our proposed algorithm), if_and_eif_train_and_predict.py (The code of two comparative models including isolation forest and extended isolation foreset), and comparative_experiment_of_unspupervised_learning.py ( 9 techniques of comparative models are applied in this file including CBLOF, HBOS, FB, KNN, Average KNN, LOF, OCSVM, DeepSVDD, VAE). The file of supervised machine learning methods applied in section 4 is comparative_experiment_of_supervised_learning.py, among which RF, XGBoost, LGBM are applied for the comparision. For all of the methods executed in section 4, the labeled_transaction_data.csv and feature_data_of_transactions.csv are required to load before execution. In addition, the files of graph convolutional network are also presented in folder of scripts.

## Requirements

* pandas 1.1.5
* numpy 1.19.2
* scikit-learn 0.23.2
* pyod 0.9.3
* xgboost 1.5.2
* lightgbm 3.1.1
* stellargraph 1.2.1
