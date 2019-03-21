from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.datasets import mnist
from keras.utils import np_utils

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
    train, test = train_test_split(data, test_size=0.3)
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
    model.add(Dense({{choice([256, 512, 1024])}}, input_shape=(7,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0, 1)}}))

    # If we choose 'four', add an additional fourth layer
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense({{choice([256, 512, 1024])}}))
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size={{choice([5, 10, 15])}},
              nb_epoch=20,
              verbose=2,
              validation_split=0.3)
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=trials)
    for trial in trials:
        print(trial)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
