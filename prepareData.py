import os
import ast
import wfdb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras as keras
from keras.utils import plot_model

from model import RNNModel

sns.set_style('darkgrid')

PATH_TO_DATA = 'E:/Датасеты ЭКГ/PTB-XL ECG dataset/archive (2)/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/'

ECG_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'ptbxl_database.csv'), index_col='ecg_id')
ECG_df.scp_codes = ECG_df.scp_codes.apply(lambda x: ast.literal_eval(x))
ECG_df.patient_id = ECG_df.patient_id.astype(int)
ECG_df.nurse = ECG_df.nurse.astype('Int64')
ECG_df.site = ECG_df.site.astype('Int64')
ECG_df.validated_by = ECG_df.validated_by.astype('Int64')

SCP_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'scp_statements.csv'), index_col=0)
SCP_df = SCP_df[SCP_df.diagnostic == 1]

def diagnostic_class(scp):
    res = set()
    for k in scp.keys():
        if k in SCP_df.index:
            res.add(SCP_df.loc[k].diagnostic_class)
    return list(res)


ECG_df['scp_classes'] = ECG_df.scp_codes.apply(diagnostic_class)

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

sampling_rate = 100

ECG_data = load_raw_data(ECG_df, sampling_rate, PATH_TO_DATA)

print(ECG_data.shape)

sample = ECG_data[0]
bar, axes = plt.subplots(sample.shape[1], 1, figsize=(20,10))
for i in range(sample.shape[1]):
    sns.lineplot(x=np.arange(sample.shape[0]), y=sample[:, i], ax=axes[i])
# plt.tight_layout()
plt.show()



import missingno as msno

msno.matrix(ECG_df)
plt.show()

ECG_df[[col for col in ECG_df.columns if col not in ('scp_codes', 'scp_classes')]].nunique(dropna=True)



X = pd.DataFrame(index=ECG_df.index)

X['age'] = ECG_df.age
X.age.fillna(0, inplace=True)

X['sex'] = ECG_df.sex.astype(float)
X.sex.fillna(0, inplace=True)

X['height'] = ECG_df.height
X.loc[X.height < 50, 'height'] = np.nan
X.height.fillna(0, inplace=True)

X['weight'] = ECG_df.weight
X.weight.fillna(0, inplace=True)

X['infarction_stadium1'] = ECG_df.infarction_stadium1.replace({
    'unknown': 0,
    'Stadium I': 1,
    'Stadium I-II': 2,
    'Stadium II': 3,
    'Stadium II-III': 4,
    'Stadium III': 5
}).fillna(0)

X['infarction_stadium2'] = ECG_df.infarction_stadium2.replace({
    'unknown': 0,
    'Stadium I': 1,
    'Stadium II': 2,
    'Stadium III': 3
}).fillna(0)

X['pacemaker'] = (ECG_df.pacemaker == 'ja, pacemaker').astype(float)



Z = pd.DataFrame(0, index=ECG_df.index, columns=['NORM', 'MI', 'STTC', 'CD', 'HYP'], dtype='int')
for i in Z.index:
    for k in ECG_df.loc[i].scp_classes:
        Z.loc[i, k] = 1


X_train, Y_train, Z_train = X[ECG_df.strat_fold <= 8],  ECG_data[X[ECG_df.strat_fold <= 8].index - 1],  Z[ECG_df.strat_fold <= 8]
X_valid, Y_valid, Z_valid = X[ECG_df.strat_fold == 9],  ECG_data[X[ECG_df.strat_fold == 9].index - 1],  Z[ECG_df.strat_fold == 9]
X_test,  Y_test,  Z_test  = X[ECG_df.strat_fold == 10], ECG_data[X[ECG_df.strat_fold == 10].index - 1], Z[ECG_df.strat_fold == 10]

print(X_train.shape, Y_train.shape, Z_train.shape)
print(X_valid.shape, Y_valid.shape, Z_valid.shape)
print(X_test.shape,  Y_test.shape,  Z_test.shape)

from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_scaler.fit(X_train)

X_train = pd.DataFrame(X_scaler.transform(X_train), columns=X_train.columns)
X_valid = pd.DataFrame(X_scaler.transform(X_valid), columns=X_valid.columns)
X_test  = pd.DataFrame(X_scaler.transform(X_test),  columns=X_test.columns)


Y_scaler = StandardScaler()
Y_scaler.fit(Y_train.reshape(-1, Y_train.shape[-1]))

Y_train = Y_scaler.transform(Y_train.reshape(-1, Y_train.shape[-1])).reshape(Y_train.shape)
Y_valid = Y_scaler.transform(Y_valid.reshape(-1, Y_valid.shape[-1])).reshape(Y_valid.shape)
Y_test  = Y_scaler.transform(Y_test.reshape(-1, Y_test.shape[-1])).reshape(Y_test.shape)


NUMPY_DATA_FILE = 'data.npz'

save_args = {
    'X_train': X_train.to_numpy().astype('float32'),
    'X_valid': X_valid.to_numpy().astype('float32'),
    'X_test':  X_test.to_numpy().astype('float32'),
    'Y_train': Y_train.astype('float32'),
    'Y_valid': Y_valid.astype('float32'),
    'Y_test':  Y_test.astype('float32'),
    'Z_train': Z_train.to_numpy().astype('float32'),
    'Z_valid': Z_valid.to_numpy().astype('float32'),
    'Z_test':  Z_test.to_numpy().astype('float32'),
}
np.savez(NUMPY_DATA_FILE, **save_args)


z1 = np.zeros(Z_test.shape)
z1[:, 0] = 1

m = keras.metrics.BinaryAccuracy()
m.update_state(Z_test, z1)
m.result()

z_prob = Z_train.sum(axis=0) / Z_train.shape[0]
z2 = np.random.uniform(size=Z_test.shape)

for i in range(z2.shape[-1]):
    z2[:, i] = (z2[:, i] < z_prob[i]).astype('float64')

m = keras.metrics.BinaryAccuracy()
m.update_state(Z_test, z2)
m.result()
#
#
# def create_model01(X_shape, Z_shape):
#     X_inputs = keras.Input(X_shape[1:], name='X_inputs')
#
#     X = create_X_model(X_inputs)
#     X = keras.layers.Dense(64, activation='relu', name='Z_dense_1')(X)
#     X = keras.layers.Dense(64, activation='relu', name='Z_dense_2')(X)
#     X = keras.layers.Dropout(0.5, name='Z_drop_1')(X)
#     outputs = keras.layers.Dense(Z_shape[-1], activation='sigmoid', name='Z_outputs')(X)
#
#     model = keras.Model(inputs=X_inputs, outputs=outputs, name='model01')
#     return model
#
#
# model01 = create_model01(X_train.shape, Z_train.shape)
# model01.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'Precision', 'Recall'])
# model01.summary()
#
# MODEL_CHECKPOINT = 'model01.keras'
#
# callbacks_list = [
#     keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=10),
#     keras.callbacks.ModelCheckpoint(filepath=MODEL_CHECKPOINT, monitor='val_binary_accuracy', save_best_only=True)
# ]
#
# history = model01.fit(X_train, Z_train, epochs=40, batch_size=32, callbacks=callbacks_list, validation_data=(X_valid, Z_valid))
#
# model01 = keras.models.load_model(MODEL_CHECKPOINT)
#
# sns.relplot(data=pd.DataFrame(history.history), kind='line', height=4, aspect=4)
# plt.show()
#
# model01.evaluate(X_test, Z_test)


# def create_Y_model(X, *, filters=(32, 64, 128), kernel_size=(5, 3, 3), strides=(1, 1, 1)):
#     f1, f2, f3 = filters
#     k1, k2, k3 = kernel_size
#     s1, s2, s3 = strides
#
#     X = keras.layers.Conv1D(f1, k1, strides=s1, padding='same', name='Y_conv_1')(X)
#     X = keras.layers.BatchNormalization(name='Y_norm_1')(X)
#     X = keras.layers.ReLU(name='Y_relu_1')(X)
#
#     X = keras.layers.MaxPool1D(2, name='Y_pool_1')(X)
#
#     X = keras.layers.Conv1D(f2, k2, strides=s2, padding='same', name='Y_conv_2')(X)
#     X = keras.layers.BatchNormalization(name='Y_norm_2')(X)
#     X = keras.layers.ReLU(name='Y_relu_2')(X)
#
#     X = keras.layers.MaxPool1D(2, name='Y_pool_2')(X)
#
#     X = keras.layers.Conv1D(f3, k3, strides=s3, padding='same', name='Y_conv_3')(X)
#     X = keras.layers.BatchNormalization(name='Y_norm_3')(X)
#     X = keras.layers.ReLU(name='Y_relu_3')(X)
#
#     X = keras.layers.GlobalAveragePooling1D(name='Y_aver')(X)
#     X = keras.layers.Dropout(0.5, name='Y_drop')(X)
#
#     return X
#
# def create_model02(X_shape, Y_shape, Z_shape):
#     X_inputs = keras.Input(X_shape[1:], name='X_inputs')
#     Y_inputs = keras.Input(Y_shape[1:], name='Y_inputs')
#
#     X = keras.layers.Concatenate(name='Z_concat')([create_X_model(X_inputs), create_Y_model(Y_inputs, filters=(64, 128, 256), kernel_size=(7, 3, 3))])
#     X = keras.layers.Dense(64, activation='relu', name='Z_dense_1')(X)
#     X = keras.layers.Dense(64, activation='relu', name='Z_dense_2')(X)
#     X = keras.layers.Dropout(0.5, name='Z_drop_1')(X)
#     outputs = keras.layers.Dense(Z_shape[-1], activation='sigmoid', name='Z_outputs')(X)
#
#     model = keras.Model(inputs=[X_inputs, Y_inputs], outputs=outputs, name='model02')
#     return model

model = RNNModel(X_train.shape, Y_train.shape, Z_train.shape)
# model02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'Precision', 'Recall'])
MODEL_CHECKPOINT = 'model02.keras'

callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=5,restore_best_weights = True, verbose = 1),
    keras.callbacks.ModelCheckpoint(filepath=MODEL_CHECKPOINT, monitor='val_binary_accuracy', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.2,  patience=5, min_lr=0.001)
]
model.train(X_train, Y_train, Z_train,callbacks_list,X_valid, Y_valid, Z_valid)
# model = keras.models.load_model(MODEL_CHECKPOINT)
plot_model(model.model, to_file='model.png')
sns.relplot(data=pd.DataFrame(model.model.history.history), kind='line', height=4, aspect=4)
plt.show()
model.evaluate(X_test, Y_test, Z_test)

layer_outputs = [layer.output for layer in model.model.layers[:5]]
activations = model.model.predict([X_test, Y_test])
first_layer_activation = activations[0]
print(first_layer_activation.shape)
plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')