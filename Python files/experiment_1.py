from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.utils import np_utils
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras import regularizers

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical


def data():
    data = pd.read_csv('facies_vectors.csv')
    feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS']
    data = data.fillna(data['PE'].mean())
    data = data.sample(frac=1).reset_index(drop=True)
    train, test = train_test_split(data, test_size=0.2)
    X_train = train[feature_names].values
    y_train = train['Facies'].values
    X_test = test[feature_names].values
    y_test = test['Facies'].values
    well_train = train['Well Name'].values
    well_test = test['Well Name'].values
    depth_train = train['Depth'].values
    depth_test = test['Depth'].values
    robust = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_train)
    X_train_robust = robust.transform(X_train)
    X_test_robust = robust.transform(X_test)
    scaler = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_train_robust)
    X_train = scaler.transform(X_train_robust)
    X_test = scaler.transform(X_test_robust)
    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)
    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):

    model = Sequential()
    model.add(Dense({{choice([512])}}, input_shape=(7,), kernel_regularizer={{choice([regularizers.l1_l2(l1=0.01, l2=0.01), regularizers.l1_l2(l1=0.001, l2=0.001), regularizers.l1_l2(l1=0.0001, l2=0.0001), regularizers.l1_l2(l1=0.00001, l2=0.00001), regularizers.l1_l2(l1=0.000001, l2=0.000001)])}},
                                                            bias_regularizer=None,
                                                            activity_regularizer=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0.3, 0.8)}}))

    model.add(Dense({{choice([512])}}, kernel_regularizer={{choice([regularizers.l1_l2(l1=0.01, l2=0.01), regularizers.l1_l2(l1=0.001, l2=0.001), regularizers.l1_l2(l1=0.0001, l2=0.0001), regularizers.l1_l2(l1=0.00001, l2=0.00001), regularizers.l1_l2(l1=0.000001, l2=0.000001)])}},
                                        bias_regularizer=regularizers.l1({{choice([0.01, 0.001, 0.0001, 0.00001, 0.000001])}}),
                                        activity_regularizer=regularizers.l1({{choice([0.01, 0.001, 0.0001, 0.00001, 0.000001])}})))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0.1, 0.6)}}))

    # If we choose 'four', add an additional fourth layer
    #if {{choice(['three', 'four'])}} == 'four':
    model.add(Dense({{choice([1024])}}, kernel_regularizer=regularizers.l1({{choice([0.01, 0.001, 0.0001, 0.00001, 0.000001])}}),
                                        bias_regularizer={{choice([regularizers.l1_l2(l1=0.01, l2=0.01), regularizers.l1_l2(l1=0.001, l2=0.001), regularizers.l1_l2(l1=0.0001, l2=0.0001), regularizers.l1_l2(l1=0.00001, l2=0.00001), regularizers.l1_l2(l1=0.000001, l2=0.000001)])}},
                                        activity_regularizer=regularizers.l1({{choice([0.01, 0.001, 0.0001, 0.00001, 0.000001])}})))
    model.add(BatchNormalization())
    model.add({{choice([Activation('linear')])}})
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_acc', patience=70, verbose=0, mode='auto', min_delta=0.01)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=25, verbose=1, mode='auto', min_delta=0.0001, min_lr=0.0000001)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.000001, verbose=1, mode='auto')

    model.fit(X_train, Y_train,
              batch_size={{choice([256])}},
              nb_epoch=10000,
              verbose=2,
              callbacks=[earlyStopping, reduce_lr],
              validation_split=0.1
              )
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=15,
                                          trials=trials)
    #for trial in trials:
        #print(trial)
    print(best_run)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
