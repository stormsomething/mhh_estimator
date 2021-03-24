import os
import joblib
import awkward as ak
import numpy as np
from argparse import ArgumentParser
from bbtautau.utils import true_mhh, features_table
from bbtautau import log; log = log.getChild('fitter')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--library', default='scikit', choices=['scikit', 'keras'])
    parser.add_argument('--fit', default=False, action='store_true')
    parser.add_argument('--gridsearch', default=False, action='store_true')
    args = parser.parse_args()

    # use all root files by default
    max_files = None
    if args.debug:
        max_files = 1

    log.info('loading samples ..')
    from bbtautau.database import dihiggs_01, dihiggs_10
    dihiggs_01.process(verbose=True, max_files=max_files)
    dihiggs_10.process(verbose=True, max_files=max_files)
    log.info('..done')
    
    if not args.fit:
        log.info('loading regressor weights')
        if args.library == 'scikit':
            regressor = joblib.load('cache/latest_scikit.clf')
        elif args.library == 'keras':
            from keras.models import load_model
            regressor = load_model('cache/latest_keras.h5')
            regressor.summary()
        else:
            pass
    else:
        log.info('prepare training data')
        train_target = ak.concatenate([
            ak.flatten(true_mhh(dihiggs_01.fold_0_array)),
            ak.flatten(true_mhh(dihiggs_10.fold_0_array))])
        train_features = np.concatenate([
            features_table(dihiggs_01.fold_0_array),
            features_table(dihiggs_10.fold_0_array)])

        if args.library == 'scikit':
            if args.gridsearch:
                from sklearn.model_selection import GridSearchCV
                from sklearn.ensemble import GradientBoostingRegressor
                # running time is prohibitive on laptop
                parameters = {
                    'n_estimators': [100, 200, 400, 600, 800, 1000, 2000],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 4, 5, 6, 7, 8],
                    'loss': ['ls'],
                }
                gbr = GradientBoostingRegressor()
                regressor_cv = GridSearchCV(gbr, parameters, n_jobs=4, verbose=True)
                log.info('fitting')
                regressor_cv.fit(train_features, train_target)
                regressor = regressor_cv.best_estimator_
                joblib.dump(regressor, 'cache/best_scikit_gridsearch.clf')
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                regressor = GradientBoostingRegressor(
                    n_estimators=2000,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=0,
                    loss='ls',
                    verbose=True)
                
                log.info('fitting')
                regressor.fit(train_features, train_target)
                joblib.dump(regressor, 'cache/latest_scikit.clf')
        elif args.library == 'keras':
            from bbtautau.models import keras_model
            regressor = keras_model((train_features.shape[1],))
            _epochs = 200
            _filename = 'cache/latest_keras.h5'
            from keras.callbacks import EarlyStopping, ModelCheckpoint
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                train_features, train_target, test_size=0.1, random_state=42)
            y_train = ak.to_numpy(y_train)
            y_test = ak.to_numpy(y_test)
            try:
                regressor.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
                history = regressor.fit(
                    X_train, y_train,
                    epochs=_epochs,
                    batch_size=50,
                    shuffle=True,
                    # validation_split=0.1,
                    validation_data=(X_test, y_test),
                    callbacks=[
                        EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                        ModelCheckpoint(
                            _filename, monitor='val_loss', 
                            verbose=True, save_best_only=True)
                    ])
                regressor.save(_filename)
                from bbtautau.plotting import nn_history
                for k in history.history.keys():
                    if 'val' in k:
                        continue
                    nn_history(history, metric=k)
                    
            except KeyboardInterrupt:
                log.info('Ended early..')
        else:
            pass
            

    log.info('plotting')
    features_test_HH_01 = features_table(dihiggs_01.fold_1_array)
    predictions_HH_01 = regressor.predict(features_test_HH_01)
    test_target_HH_01 = ak.flatten(true_mhh(dihiggs_01.fold_1_array))

    features_test_HH_10 = features_table(dihiggs_10.fold_1_array)
    predictions_HH_10 = regressor.predict(features_test_HH_10)
    test_target_HH_10 = ak.flatten(true_mhh(dihiggs_10.fold_1_array))

    if args.library == 'keras':
        predictions_HH_01 = np.reshape(
            predictions_HH_01, (predictions_HH_01.shape[0], ))
        predictions_HH_10 = np.reshape(
            predictions_HH_10, (predictions_HH_10.shape[0], ))
    
    from bbtautau.plotting import signal_pred_target_comparison, signal_features
    # 
    signal_pred_target_comparison(
        predictions_HH_10, predictions_HH_01,
        test_target_HH_10, test_target_HH_01,
        dihiggs_10, dihiggs_01,
        regressor=args.library)
    # 
    signal_features(dihiggs_01, dihiggs_10)


