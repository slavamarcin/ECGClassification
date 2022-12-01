import os

from keras import Input, Model
from keras.layers import Dropout, Dense, GRU, Embedding, Flatten, Bidirectional, concatenate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

class SequentialModel():
    def __init__(self):
        # self.model = keras.Sequential([
        #     Input(shape=(12, 1000)),
        #     Flatten(),
        #     # Input(shape=[None,12000]),
        #     # Dense(4096, activation='relu'),
        #     Dense(2048, activation='relu'),
        #     Dense(1024, activation='relu'),
        #     Dropout(0.4),
        #     Dense(512, activation='relu'),
        #     Dense(256, activation='sigmoid'),
        #     Dense(5, activation='softmax')
        # ])
        # define two sets of inputs
        input1 = Input(shape=(None,1000))
        input2 = Input(shape=(None,1000))
        input3 = Input(shape=(None,1000))
        input4 = Input(shape=(None,1000))
        input5 = Input(shape=(None,1000))
        input6 = Input(shape=(None,1000))
        input7 = Input(shape=(None,1000))
        input8 = Input(shape=(None,1000))
        input9 = Input(shape=(None,1000))
        input10 = Input(shape=(None,1000))
        input11 = Input(shape=(None,1000))
        input12 = Input(shape=(None,1000))

        # the first branch operates on the first input
        x = Dense(1024, activation="relu")(input1)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.2)(x)
        # x = Dense(5, activation="relu")(x)
        x = Model(inputs=input1, outputs=x)

        # the second branch opreates on the second input
        q = Dense(1024, activation="relu")(input2)
        q = Dense(512, activation="relu")(q)
        q = Dropout(0.2)(q)
        # q = Dense(5, activation="relu")(q)
        q = Model(inputs=input2, outputs=q)

        w = Dense(1024, activation="relu")(input3)
        w = Dense(512, activation="relu")(w)
        w = Dropout(0.2)(w)
        # w = Dense(5, activation="relu")(w)
        w = Model(inputs=input3, outputs=w)

        e = Dense(1024, activation="relu")(input4)
        e = Dense(512, activation="relu")(e)
        e = Dropout(0.2)(e)
        # e = Dense(5, activation="relu")(e)
        e = Model(inputs=input4, outputs=e)

        r = Dense(1024, activation="relu")(input5)
        r = Dense(512, activation="relu")(r)
        r = Dropout(0.2)(r)
        # r = Dense(5, activation="relu")(r)
        r = Model(inputs=input5, outputs=r)

        t = Dense(1024, activation="relu")(input6)
        t = Dense(512, activation="relu")(t)
        t = Dropout(0.2)(t)
        # t = Dense(5, activation="relu")(t)
        t = Model(inputs=input6, outputs=t)

        y = Dense(1024, activation="relu")(input7)
        y = Dense(512, activation="relu")(y)
        y = Dropout(0.2)(y)
        # y = Dense(5, activation="relu")(y)
        y = Model(inputs=input7, outputs=y)

        o = Dense(1024, activation="relu")(input8)
        o = Dense(512, activation="relu")(o)
        o = Dropout(0.2)(o)
        # o = Dense(5, activation="relu")(o)
        o = Model(inputs=input8, outputs=o)

        u = Dense(1024, activation="relu")(input9)
        u = Dense(512, activation="relu")(u)
        u = Dropout(0.2)(u)
        # u = Dense(5, activation="relu")(u)
        u = Model(inputs=input9, outputs=u)

        i = Dense(1024, activation="relu")(input10)
        i = Dense(512, activation="relu")(i)
        i = Dropout(0.2)(i)
        # i = Dense(5, activation="relu")(i)
        i = Model(inputs=input10, outputs=i)

        p = Dense(1024, activation="relu")(input11)
        p = Dense(512, activation="relu")(p)
        p = Dropout(0.2)(p)
        # p = Dense(5, activation="relu")(p)
        p = Model(inputs=input11, outputs=p)

        a = Dense(1024, activation="relu")(input12)
        a = Dense(512, activation="relu")(a)
        a = Dropout(0.2)(a)
        # a = Dense(5, activation="relu")(a)
        a = Model(inputs=input12, outputs=a)

        # combine the output of the two branches
        combined = concatenate([x.output, q.output,w.output, e.output,r.output, t.output,y.output, u.output,i.output, o.output,p.output, a.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(126, activation="relu")(combined)
        z = Dense(5, activation="softmax")(z)

        # our model will accept the inputs of the two branches and
        # then output a single value
        self.model = Model(inputs=[x.input, q.input, w.input, e.input, r.input, t.input, y.input, u.input, i.input, o.input, p.input, a.input], outputs=z)

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
    def __init__(self,X_train):
        self.model = keras.Sequential([
            # Embedding(10000, 1000,input_shape=(X_train.shape[1],X_train.shape[2])),
            Input(shape=(X_train.shape[1],X_train.shape[2])),
            # Flatten(),
            # Embedding(input_dim=12000, output_dim=64),
            Bidirectional(GRU(256, return_sequences=True)),
            Bidirectional(GRU(128)),
            Dense(1024, activation='sigmoid'),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(5, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model.summary()
    def train(self,x_train,y_train,x_test,y_test):
        self.model.fit(x = x_train,y = y_train, validation_data=(x_test,y_test), batch_size=56, epochs=10, verbose=1)
        print('\nhistory dict:', self.model.history)
        self.model.save('modelRNN')

    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test)

    def predict(self,data):
        return self.model.predict(data)