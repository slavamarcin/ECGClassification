import pandas as pd
import numpy as np
import wfdb
import ast

from model import SequentialModel

def preprocess(y,x):
    listofy = []
    listofx = []
    for i, y in enumerate(y):
        if len(y) == 1:
            listofy.append(y[0])
            listofx.append(x[i])
    return listofy,listofx

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = 'E:/Датасеты ЭКГ/PTB-XL ECG dataset/archive (2)/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
x_train = np.array(X_train)
y_train = np.array(y_train)
x_test = np.array(X_test)
y_test = np.array(y_test)
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()

listofy,listofx = preprocess(y_train,x_train)
y_train = np.array(listofy)
x_train = np.array(listofx)

listofy,listofx = preprocess(y_test,x_test)
y_test = np.array(listofy)
x_test = np.array(listofx)

y_train = onehotencoder.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = onehotencoder.transform(y_test.reshape(-1, 1)).toarray()

model = SequentialModel()
model.train(x_train = x_train[::,::,1],y_train =y_train,x_test = x_test[::,::,1],y_test = y_test)
pass

