import os

from keras import Input, Model, regularizers
from keras.layers import Dropout, Dense, GRU, Embedding, Flatten, Bidirectional, concatenate, Conv1D, Reshape, \
    BatchNormalization, ReLU, Conv2D, Add, AveragePooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras, Tensor


class SequentialModel():
    def __init__(self,X_shape,Y_shape,Z_shape):
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
        input13 = Input(shape=(None,7))

        # the first branch operates on the first input
        x = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input1)
        x = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(x)
        x = Dropout(0.3)(x)
        x = Model(inputs=input1, outputs=x)

        # the second branch opreates on the second input
        q = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input2)
        q = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(q)
        q = Dropout(0.3)(q)
        q = Model(inputs=input2, outputs=q)

        w = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input3)
        w = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(w)
        w = Dropout(0.3)(w)
        w = Model(inputs=input3, outputs=w)

        e = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input4)
        e = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(e)
        e = Dropout(0.3)(e)
        e = Model(inputs=input4, outputs=e)

        r = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input5)
        r = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(r)
        r = Dropout(0.3)(r)
        r = Model(inputs=input5, outputs=r)

        t = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input6)
        t = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(t)
        t = Dropout(0.3)(t)
        t = Model(inputs=input6, outputs=t)

        y = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input7)
        y = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(y)
        y = Dropout(0.2)(y)
        y = Model(inputs=input7, outputs=y)

        o = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input8)
        o = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(o)
        o = Dropout(0.2)(o)
        o = Model(inputs=input8, outputs=o)

        u = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input9)
        u = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(u)
        u = Dropout(0.2)(u)
        u = Model(inputs=input9, outputs=u)

        i = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input10)
        i = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(i)
        i = Dropout(0.2)(i)
        i = Model(inputs=input10, outputs=i)

        p = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input11)
        p = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(p)
        p = Dropout(0.2)(p)
        p = Model(inputs=input11, outputs=p)

        a = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input12)
        a = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(a)
        a = Dropout(0.2)(a)
        a = Model(inputs=input12, outputs=a)

        s = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input13)
        s = Dense(256, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(s)
        s = Dropout(0.3)(s)
        s = Model(inputs=input13, outputs=s)

        # combine the output of the two branches
        combined = concatenate([x.output, q.output,w.output, e.output,r.output, t.output,y.output, u.output,i.output, o.output,p.output, a.output,s.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(126, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(combined)
        outputs = keras.layers.Dense(Z_shape[-1], activation='sigmoid', name='Z_outputs',kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(z)

        # our model will accept the inputs of the two branches and
        # then output a single value
        self.model = Model(inputs=[x.input, q.input, w.input, e.input, r.input, t.input, y.input, u.input, i.input, o.input, p.input, a.input,s.input], outputs=outputs)

        self.model.compile(optimizer='adam',
                       loss='binary_crossentropy', metrics=['binary_accuracy', 'Precision', 'Recall'])
        self.model.summary()
    def train(self,x_train,y_train,x_test,y_test,callbacks_list):
        self.model.fit(x = x_train, y = y_train, validation_data=(x_test,y_test), callbacks=callbacks_list, batch_size=56, epochs=10, verbose=1)
        print('\nhistory dict:', self.model.history)
        self.model.save('model')

    def evaluate(self,x_test,y_test):
        return self.model.evaluate(x_test,y_test)

    def predict(self,data):
        return self.model.predict(data)


class RNNModel:

    def relu_bn(inputs: Tensor) -> Tensor:
        relu = ReLU()(inputs)
        bn = BatchNormalization()(relu)
        return bn

    def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
        y = Conv2D(kernel_size=kernel_size,
                   strides=(1 if not downsample else 2),
                   filters=filters,
                   padding="same")(x)
        y = RNNModel.relu_bn(y)
        y = Conv2D(kernel_size=kernel_size,
                   strides=1,
                   filters=filters,
                   padding="same")(y)

        if downsample:
            x = Conv2D(kernel_size=1,
                       strides=2,
                       filters=filters,
                       padding="same")(x)
        out = Add()([x, y])
        out = RNNModel.relu_bn(out)
        return out

    def create_X_model(self,X, *, dropouts=0.3):
        X = keras.layers.Dense(512, activation='relu', name='X_dense_1')(X)
        X = keras.layers.Dropout(dropouts, name='X_drop_1')(X)
        X = keras.layers.Dense(256, activation='relu', name='X_dense_2')(X)
        X = keras.layers.Dropout(dropouts, name='X_drop_2')(X)

        return X

    def __init__(self,X_shape,Y_shape,Z_shape):
        X_inputs = keras.Input(X_shape[1:], name='X_inputs')
        Y_inputs = keras.Input(Y_shape[1:], name='Y_inputs')
        ##---------------------------CRNN----------------------------------------------
        # input1 = Input(shape=(Y_inputs.shape[1],Y_inputs.shape[2]))
        x = Conv1D(64,kernel_size=4, activation='relu',padding="same")(Y_inputs)
        x = Conv1D(64, kernel_size=4, activation='relu',padding="same") (x)
        x = keras.layers.MaxPool1D(4,padding="same") (x)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=4, activation='relu',padding="same")(x)
        x = Conv1D(128, kernel_size=4, activation='relu',padding="same")(x)
        x = keras.layers.MaxPool1D(6,padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1D(256, kernel_size=4, activation='relu',padding="same")(x)
        x = Conv1D(256, kernel_size=4, activation='relu',padding="same")(x)
        x = keras.layers.MaxPool1D(6,padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1D(512, kernel_size=4, activation='relu',padding="same")(x)
        x = Conv1D(512, kernel_size=4, activation='relu',padding="same")(x)
        x = keras.layers.MaxPool1D(8,padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(GRU(128, return_sequences=True))(x)
        x = Bidirectional(GRU(64))(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        # x = keras.layers.Concatenate(name='Z_concat')(
        #     [self.create_X_model(X_inputs), x])

        outputs = keras.layers.Dense(Z_shape[-1], activation='sigmoid', name='Z_outputs')(x)
        self.model = keras.Model(inputs=[X_inputs, Y_inputs], outputs=outputs, name='modelRNN')
        self.model.compile(optimizer='adam',
                       loss='binary_crossentropy', metrics=['binary_accuracy', 'Precision', 'Recall'])

        self.model.summary()
    def train(self,X_train,Y_train,Z_train,callbacks_list,X_valid,Y_valid,Z_valid):
        self.model.fit([X_train, Y_train], Z_train, epochs=10, batch_size=32, callbacks=callbacks_list, validation_data=([X_valid, Y_valid], Z_valid))
        print('\nhistory dict:', self.model.history)
        self.model.save('modelRNN')

    def evaluate(self,X_test, Y_test, Z_test):
        return self.model.evaluate([X_test, Y_test], Z_test)

    def predict(self,data):
        return self.model.predict(data)