import os

from keras import Input
from keras.layers import Dropout, Dense, GRU

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

class SequentialModel():
    def __init__(self):
        self.model = keras.Sequential([
            Input(shape=[None, 1000]),
            Dense(2048, activation='relu'),
            Dense(1024, activation='relu'),
            Dropout(0.4),
            Dense(512, activation='relu'),
            Dense(5, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model.summary()
    def train(self,x_train,y_train,x_test,y_test):
        self.model.fit(x = x_train, y = y_train, validation_data=(x_test,y_test), batch_size=56, epochs=10, verbose=1)
        print('\nhistory dict:', self.model.history)
        self.model.save('model')

    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test)

    def predict(self,data):
        return self.model.predict(data)


class RNNModel:
    def __init__(self):
        self.model = keras.Sequential([
            Input(shape=(1000,12)),
            GRU(256, return_sequences=True),
            GRU(128, return_sequences=True),
            Dense(1024, activation='relu'),
            Dropout(0.3),
            Dense(5, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model.summary()
    def train(self,x_train,y_train,x_test,y_test):
        self.model.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=56, epochs=2, verbose=1)
        print('\nhistory dict:', self.model.history)
        self.model.save('model')

    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test)

    def predict(self,data):
        return self.model.predict(data)