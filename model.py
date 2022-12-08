import os

from keras import Input, Model, regularizers
from keras.layers import Dropout, Dense, GRU, Embedding, Flatten, Bidirectional, concatenate, Conv1D, Reshape, \
    BatchNormalization, ReLU, Conv2D, Add, AveragePooling2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras, Tensor


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
        x = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input1)
        x = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(x)
        x = Dropout(0.2)(x)
        # x = Dense(5, activation="relu")(x)
        x = Model(inputs=input1, outputs=x)

        # the second branch opreates on the second input
        q = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input2)
        q = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(q)
        q = Dropout(0.2)(q)
        # q = Dense(5, activation="relu")(q)
        q = Model(inputs=input2, outputs=q)

        w = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input3)
        w = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(w)
        w = Dropout(0.2)(w)
        # w = Dense(5, activation="relu")(w)
        w = Model(inputs=input3, outputs=w)

        e = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input4)
        e = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(e)
        e = Dropout(0.2)(e)
        # e = Dense(5, activation="relu")(e)
        e = Model(inputs=input4, outputs=e)

        r = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input5)
        r = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(r)
        r = Dropout(0.2)(r)
        # r = Dense(5, activation="relu")(r)
        r = Model(inputs=input5, outputs=r)

        t = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input6)
        t = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(t)
        t = Dropout(0.2)(t)
        # t = Dense(5, activation="relu")(t)
        t = Model(inputs=input6, outputs=t)

        y = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input7)
        y = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(y)
        y = Dropout(0.2)(y)
        # y = Dense(5, activation="relu")(y)
        y = Model(inputs=input7, outputs=y)

        o = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input8)
        o = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(o)
        o = Dropout(0.2)(o)
        # o = Dense(5, activation="relu")(o)
        o = Model(inputs=input8, outputs=o)

        u = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input9)
        u = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(u)
        u = Dropout(0.2)(u)
        # u = Dense(5, activation="relu")(u)
        u = Model(inputs=input9, outputs=u)

        i = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input10)
        i = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(i)
        i = Dropout(0.2)(i)
        # i = Dense(5, activation="relu")(i)
        i = Model(inputs=input10, outputs=i)

        p = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input11)
        p = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(p)
        p = Dropout(0.2)(p)
        # p = Dense(5, activation="relu")(p)
        p = Model(inputs=input11, outputs=p)

        a = Dense(1024, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(input12)
        a = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(a)
        a = Dropout(0.2)(a)
        # a = Dense(5, activation="relu")(a)
        a = Model(inputs=input12, outputs=a)

        # combine the output of the two branches
        combined = concatenate([x.output, q.output,w.output, e.output,r.output, t.output,y.output, u.output,i.output, o.output,p.output, a.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(126, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(combined)
        z = Dense(5, activation="softmax",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.L2(1e-4),
        activity_regularizer=regularizers.L2(1e-5))(z)

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


    def __init__(self,X_train):
        #---------------------------ResNet----------------------------------------------
        # num_filters = 64
        # inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
        # t = BatchNormalization()(inputs)
        # t = Conv2D(kernel_size=3,
        #            strides=1,
        #            filters=num_filters,
        #            padding="same")(t)
        # t = self.relu_bn(t)
        #
        # num_blocks_list = [2, 5, 5, 2]
        # for i in range(len(num_blocks_list)):
        #     num_blocks = num_blocks_list[i]
        #     for j in range(num_blocks):
        #         t = self.residual_block(t, downsample=(j == 0 and i != 0), filters=num_filters)
        #     num_filters *= 2
        #
        # t = AveragePooling2D(4)(t)
        # t = Flatten()(t)
        # outputs = Dense(5, activation='softmax')(t)
        #
        # self.model = Model(inputs, outputs)
        #
        # self.model.compile(
        #     optimizer='adam',
        #     loss='sparse_categorical_crossentropy',
        #     metrics=['accuracy']
        # )


        ##---------------------------CRNN----------------------------------------------

        self.model = keras.Sequential()
        input1 = Input(shape=(X_train.shape[1],X_train.shape[2]))
        x = Conv1D(64,kernel_size=4, activation='relu',padding="same")(input1)
        x = Conv1D(64, kernel_size=4, activation='relu',padding="same") (x)
        x = keras.layers.MaxPool1D(8,padding="same") (x)
        x = BatchNormalization()(x)
        x = Conv1D(128, kernel_size=4, activation='relu',padding="same")(x)
        x = Conv1D(128, kernel_size=4, activation='relu',padding="same")(x)
        x = keras.layers.MaxPool1D(8,padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1D(256, kernel_size=4, activation='relu',padding="same")(x)
        x = Conv1D(256, kernel_size=4, activation='relu',padding="same")(x)
        x = keras.layers.MaxPool1D(8,padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv1D(512, kernel_size=4, activation='relu',padding="same")(x)
        x = Conv1D(512, kernel_size=4, activation='relu',padding="same")(x)
        x = keras.layers.MaxPool1D(8,padding="same")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
            # Reshape((12000, 256), input_shape=(None,12000, 256)),
        x = Bidirectional(GRU(128, return_sequences=True))(x)
        x = Bidirectional(GRU(64))(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(5, activation='softmax')(x)
        self.model = Model(inputs=input1, outputs=x)



        ## -----------------------------------------------------------
        # input1 = Input(shape=(None,1000))
        # input2 = Input(shape=(None,1000))
        # input3 = Input(shape=(None,1000))
        # input4 = Input(shape=(None,1000))
        # input5 = Input(shape=(None,1000))
        # input6 = Input(shape=(None,1000))
        # input7 = Input(shape=(None,1000))
        # input8 = Input(shape=(None,1000))
        # input9 = Input(shape=(None,1000))
        # input10 = Input(shape=(None,1000))
        # input11 = Input(shape=(None,1000))
        # input12 = Input(shape=(None,1000))
        #
        # # the first branch operates on the first input
        # x = Bidirectional(GRU(256, return_sequences=True))(input1)
        # x = Bidirectional(GRU(128)),(x)
        # x = Dense(512, activation="relu")(x)
        # # x = Dropout(0.2)(x)
        # # x = Dense(5, activation="relu")(x)
        # x = Model(inputs=input1, outputs=x)
        #
        # # the second branch opreates on the second input
        # q = Bidirectional(GRU(256, return_sequences=True))(input1)
        # q = Bidirectional(GRU(128)),(q)
        # # q = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(q)
        # # q = Dropout(0.2)(q)
        # # q = Dense(5, activation="relu")(q)
        # q = Model(inputs=input2, outputs=q)
        #
        # w = Bidirectional(GRU(256, return_sequences=True))(input1)
        # w = Bidirectional(GRU(128)),(w)
        # # w = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(w)
        # # w = Dropout(0.2)(w)
        # # w = Dense(5, activation="relu")(w)
        # w = Model(inputs=input3, outputs=w)
        #
        # e = Bidirectional(GRU(256, return_sequences=True))(input1)
        # e = Bidirectional(GRU(128)),(e)
        # # e = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(e)
        # # e = Dropout(0.2)(e)
        # # e = Dense(5, activation="relu")(e)
        # e = Model(inputs=input4, outputs=e)
        #
        # r = Bidirectional(GRU(256, return_sequences=True))(input1)
        # r = Bidirectional(GRU(128)), (r)
        # # r = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(r)
        # # r = Dropout(0.2)(r)
        # # r = Dense(5, activation="relu")(r)
        # r = Model(inputs=input5, outputs=r)
        #
        # t = Bidirectional(GRU(256, return_sequences=True))(input1)
        # t = Bidirectional(GRU(128)),(t)
        # # t = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(t)
        # # t = Dropout(0.2)(t)
        # # t = Dense(5, activation="relu")(t)
        # t = Model(inputs=input6, outputs=t)
        #
        # y = Bidirectional(GRU(256, return_sequences=True))(input1)
        # y = Bidirectional(GRU(128)),(y)
        # # y = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(y)
        # # y = Dropout(0.2)(y)
        # # y = Dense(5, activation="relu")(y)
        # y = Model(inputs=input7, outputs=y)
        #
        # o = Bidirectional(GRU(256, return_sequences=True))(input1)
        # o = Bidirectional(GRU(128)),(o)
        # # o = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(o)
        # # o = Dropout(0.2)(o)
        # # o = Dense(5, activation="relu")(o)
        # o = Model(inputs=input8, outputs=o)
        #
        # u = Bidirectional(GRU(256, return_sequences=True))(input1)
        # u = Bidirectional(GRU(128)),(u)
        # # u = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(u)
        # # u = Dropout(0.2)(u)
        # # u = Dense(5, activation="relu")(u)
        # u = Model(inputs=input9, outputs=u)
        #
        # i = Bidirectional(GRU(256, return_sequences=True))(input1)
        # i = Bidirectional(GRU(128)),(i)
        # # i = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(i)
        # # i = Dropout(0.2)(i)
        # # i = Dense(5, activation="relu")(i)
        # i = Model(inputs=input10, outputs=i)
        #
        # p = Bidirectional(GRU(256, return_sequences=True))(input1)
        # p = Bidirectional(GRU(128)),(p)
        # # p = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(p)
        # # p = Dropout(0.2)(p)
        # # p = Dense(5, activation="relu")(p)
        # p = Model(inputs=input11, outputs=p)
        #
        # a = Bidirectional(GRU(256, return_sequences=True))(input1)
        # a = Bidirectional(GRU(128)),(a)
        # # a = Dense(512, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # # bias_regularizer=regularizers.L2(1e-4),
        # # activity_regularizer=regularizers.L2(1e-5))(a)
        # # a = Dropout(0.2)(a)
        # # a = Dense(5, activation="relu")(a)
        # a = Model(inputs=input12, outputs=a)
        #
        # # combine the output of the two branches
        # combined = concatenate([x.output, q.output,w.output, e.output,r.output, t.output,y.output, u.output,i.output, o.output,p.output, a.output])
        #
        # # apply a FC layer and then a regression prediction on the
        # # combined outputs
        # z = Dense(126, activation="relu",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # bias_regularizer=regularizers.L2(1e-4),
        # activity_regularizer=regularizers.L2(1e-5))(combined)
        # z = Dense(5, activation="softmax",kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        # bias_regularizer=regularizers.L2(1e-4),
        # activity_regularizer=regularizers.L2(1e-5))(z)
        # self.model = Model(inputs=[x.input, q.input, w.input, e.input, r.input, t.input, y.input, u.input, i.input, o.input, p.input, a.input], outputs=z)
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