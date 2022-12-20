import os
import ast
import wfdb
from gtda.plotting import plot_diagram
from keras import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import keras as keras
from keras.saving.legacy.save import load_model
from keras.utils import plot_model

from model import RNNModel, SequentialModel

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


def trainRNNmodel(X_train, Y_train, Z_train,X_test,Y_test, Z_test, X_valid, Y_valid, Z_valid):
    model = RNNModel(X_train.shape, Y_train.shape, Z_train.shape)
    # model02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'Precision', 'Recall'])
    MODEL_CHECKPOINT = 'model02.keras'

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_CHECKPOINT, monitor='val_binary_accuracy', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.2, patience=5, min_lr=0.001)
    ]
    model.train(X_train, Y_train, Z_train, callbacks_list, X_valid, Y_valid, Z_valid)
    # model = keras.models.load_model(MODEL_CHECKPOINT)
    plot_model(model.model, to_file='model.png')
    sns.relplot(data=pd.DataFrame(model.model.history.history), kind='line', height=4, aspect=4)
    plt.show()
    model.evaluate(X_test, Y_test, Z_test)


def trainSeqmodel(X_train, Y_train, Z_train,X_test, Y_test, Z_test, X_valid, Y_valid, Z_valid):
    model = SequentialModel(X_train.shape, Y_train.shape, Z_train.shape)
    # model02.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 'Precision', 'Recall'])
    MODEL_CHECKPOINT = 'model02.keras'

    callbacks_list = [
        keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_CHECKPOINT, monitor='val_binary_accuracy', save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.2, patience=5, min_lr=0.001)
    ]
    model.train(
        x_train=[Y_train[::, ::, 0], Y_train[::, ::, 1], Y_train[::, ::, 2], Y_train[::, ::, 3], Y_train[::, ::, 4],
                 Y_train[::, ::, 5], Y_train[::, ::, 6],
                 Y_train[::, ::, 7], Y_train[::, ::, 8], Y_train[::, ::, 9], Y_train[::, ::, 10], Y_train[::, ::, 11],
                 X_train], y_train=Z_train, callbacks_list=callbacks_list,
        x_test=[Y_valid[::, ::, 0], Y_valid[::, ::, 1],
                 Y_valid[::, ::, 2], Y_valid[::, ::, 3],
                 Y_valid[::, ::, 4], Y_valid[::, ::, 5],
                 Y_valid[::, ::, 6], Y_valid[::, ::, 7],
                 Y_valid[::, ::, 8], Y_valid[::, ::, 9],
                 Y_valid[::, ::, 10], Y_valid[::, ::, 11],
                 X_valid], y_test=Z_valid)
    # model = keras.models.load_model(MODEL_CHECKPOINT)
    plot_model(model.model, to_file='modelSeq.png')
    sns.relplot(data=pd.DataFrame(model.model.history.history), kind='line', height=4, aspect=4)
    plt.show()
    # model.evaluate(X_test, Y_test, Z_test)


sampling_rate = 100

ECG_data = load_raw_data(ECG_df, sampling_rate, PATH_TO_DATA)

print(ECG_data.shape)

sample = ECG_data[0]
bar, axes = plt.subplots(sample.shape[1], 1, figsize=(20, 10))
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

X_train, Y_train, Z_train = X[ECG_df.strat_fold <= 8], ECG_data[X[ECG_df.strat_fold <= 8].index - 1], Z[
    ECG_df.strat_fold <= 8]
X_valid, Y_valid, Z_valid = X[ECG_df.strat_fold == 9], ECG_data[X[ECG_df.strat_fold == 9].index - 1], Z[
    ECG_df.strat_fold == 9]
X_test, Y_test, Z_test = X[ECG_df.strat_fold == 10], ECG_data[X[ECG_df.strat_fold == 10].index - 1], Z[
    ECG_df.strat_fold == 10]

print(X_train.shape, Y_train.shape, Z_train.shape)
print(X_valid.shape, Y_valid.shape, Z_valid.shape)
print(X_test.shape, Y_test.shape, Z_test.shape)

from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_scaler.fit(X_train)

X_train = pd.DataFrame(X_scaler.transform(X_train), columns=X_train.columns)
X_valid = pd.DataFrame(X_scaler.transform(X_valid), columns=X_valid.columns)
X_test = pd.DataFrame(X_scaler.transform(X_test), columns=X_test.columns)

Y_scaler = StandardScaler()
Y_scaler.fit(Y_train.reshape(-1, Y_train.shape[-1]))

Y_train = Y_scaler.transform(Y_train.reshape(-1, Y_train.shape[-1])).reshape(Y_train.shape)
Y_valid = Y_scaler.transform(Y_valid.reshape(-1, Y_valid.shape[-1])).reshape(Y_valid.shape)
Y_test = Y_scaler.transform(Y_test.reshape(-1, Y_test.shape[-1])).reshape(Y_test.shape)

NUMPY_DATA_FILE = 'data.npz'

save_args = {
    'X_train': X_train.to_numpy().astype('float32'),
    'X_valid': X_valid.to_numpy().astype('float32'),
    'X_test': X_test.to_numpy().astype('float32'),
    'Y_train': Y_train.astype('float32'),
    'Y_valid': Y_valid.astype('float32'),
    'Y_test': Y_test.astype('float32'),
    'Z_train': Z_train.to_numpy().astype('float32'),
    'Z_valid': Z_valid.to_numpy().astype('float32'),
    'Z_test': Z_test.to_numpy().astype('float32'),
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

trainRNNmodel(X_train, Y_train, Z_train, Y_test, Z_test, X_valid, Y_valid, Z_valid)  # Обучение RNN модели
trainSeqmodel(X_train, Y_train, Z_train,X_test, Y_test, Z_test, X_valid, Y_valid, Z_valid) # Обучение полносвязной модели

model = load_model('model02.keras')
layer_outputs = [layer.output for layer in model.layers[:19]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict([X_test[0:10], Y_test[0:10]])
layer_activation = activations[0]
print(layer_activation.shape)

# plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

from gtda.homology import VietorisRipsPersistence

VR = VietorisRipsPersistence(metric="euclidean",
    homology_dimensions=[0,1,2],
    n_jobs=6,
    collapse_edges=True,
)  # Parameter explained in the text
diagrams = VR.fit_transform(layer_activation)
plot_data = plot_diagram(diagrams[0])
print("OK")
