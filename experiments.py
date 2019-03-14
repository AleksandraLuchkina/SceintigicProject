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
from keras.utils import to_categorical

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import diffind
import augmentation
def data():
	data = pd.read_csv('facies_vectors.csv')
	data['Facies'] -= 1
	feature_names = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE', 'NM_M', 'RELPOS','GR_diff_up', 'ILD_log10_diff_up', 'DeltaPHI_diff_up', 				'PHIND_diff_up', 'PE_diff_up', 'NM_M_diff_up', 'RELPOS_diff_up']
	data = data.fillna(data['PE'].mean())
	new_data = diffind.data_with_diff(data)
	test = new_data[new_data['Well Name'] == 'NEWBY']
	train = new_data[new_data['Well Name'] != 'NEWBY']
	X_train_1 = train[feature_names].values
	y_train = train['Facies'].values
	X_test_1 = test[feature_names].values
	y_test = test['Facies'].values
	well_train = train['Well Name'].values
	well_test = test['Well Name'].values
	depth_train = train['Depth'].values
	depth_test = test['Depth'].values
	X_aug_train = augmentation.augment_features(X_train_1,well_train,depth_train)
	X_aug_test = augmentation.augment_features(X_test_1,well_test,depth_test)
	robust = preprocessing.RobustScaler(quantile_range=(25.0, 75.0)).fit(X_aug_train)
	X_train_robust = robust.transform(X_aug_train)
	X_test_robust = robust.transform(X_aug_test)
	scaler = StandardScaler().fit(X_train_robust)
	X_train_robust_norm = scaler.transform(X_train_robust)
	X_test_robust_norm = scaler.transform(X_test_robust)
	X_train = X_train_robust_norm
	X_test = X_test_robust_norm
	Y_train = to_categorical(y_train, 9)
	Y_test = to_categorical(y_test, 9)
	return(X_train, Y_train, X_test, Y_test)


def model(X_train, Y_train, X_test, Y_test):

    model = Sequential()
    model.add(Dense({{choice([512])}}, input_shape=(70,), kernel_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}},
                                                            bias_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}},
                                                            activity_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}}))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0.3, 0.8)}}))

    model.add(Dense({{choice([512])}}, kernel_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}},
                                        bias_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}},
                                        activity_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}}))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid'])}}))
    model.add(Dropout({{uniform(0.1, 0.6)}}))

    # If we choose 'four', add an additional fourth layer
    #if {{choice(['three', 'four'])}} == 'four':
    model.add(Dense({{choice([1024])}}, kernel_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}},
                                        bias_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}},
                                        activity_regularizer={{choice([None, regularizers.l2(0.00001), regularizers.l1(0.00001), regularizers.l1_l2(l1=0.00001, l2=0.00001)])}}))
    model.add(BatchNormalization())
    model.add({{choice([Activation('linear')])}})
    model.add(Activation('relu'))

    model.add(Dense(9))
    model.add(Activation('softmax'))

    optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=200, verbose=0, mode='auto')
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
                                          max_evals=5,
                                          trials=trials)
    #for trial in trials:
        #print(trial)
    print(best_run)
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
