import os
import joblib
import awkward as ak
import numpy as np
from argparse import ArgumentParser
from bbtautau import log; log = log.getChild('fitter')
from bbtautau.utils import features_table, universal_true_mhh, visable_mass, clean_samples, chi_square_test, rotate_events
from bbtautau.plotting import signal_features, ztautau_pred_target_comparison, roc_plot_rnn_mmc, rnn_mmc_comparison, avg_mhh_calculation, avg_mhh_plot
from bbtautau.database import dihiggs_01, dihiggs_10, ztautau, ttbar
from bbtautau.models import keras_model_main
from bbtautau.plotting import nn_history
from bbtautau.mmc import mmc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.utils.vis_utils import plot_model

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--library', default='scikit', choices=['scikit', 'keras'])
    parser.add_argument('--fit', default=False, action='store_true')
    parser.add_argument('--gridsearch', default=False, action='store_true')
    parser.add_argument('--use-cache', default=False, action='store_true')
    args = parser.parse_args()

    # use all root files by default
    max_files = None
    if args.debug:
        max_files = 1
        if args.use_cache:
            log.info('N(files) limit is irrelevant with the cache')

    log.info('loading samples ..')

    dihiggs_01.process(
        verbose=True,
        max_files=max_files,
        use_cache=args.use_cache,
        remove_bad_training_events=True)
    dihiggs_10.process(
        verbose=True,
        max_files=max_files,
        use_cache=args.use_cache,
        remove_bad_training_events=True)
    ztautau.process(
        verbose=True,
        is_signal=False,
        max_files=max_files,
        use_cache=args.use_cache,
        remove_bad_training_events=True)
    ttbar.process(
        verbose=True,
        is_signal=False,
        max_files=max_files,
        use_cache=args.use_cache,
        remove_bad_training_events=True)
    log.info('..done')

    if not args.fit:
        log.info('loading regressor weights')
        if args.library == 'scikit':
            regressor = joblib.load('cache/latest_scikit.clf')
        elif args.library == 'keras':
            regressor = load_model('cache/my_keras_training.h5')
            # regressor = load_model('cache/best_keras_training.h5')
            regressor.summary()
        else:
            pass

    else:
        log.info('prepare training data')

        dihiggs_01_target = dihiggs_01.fold_0_array['universal_true_mhh']
        dihiggs_10_target = dihiggs_10.fold_0_array['universal_true_mhh']
        ztautau_target = ztautau.fold_0_array['universal_true_mhh']
        ttbar_target = ttbar.fold_0_array['universal_true_mhh']
        
        dihiggs_01_vis_mass = visable_mass(dihiggs_01.fold_0_array, 'dihiggs_01')
        dihiggs_10_vis_mass = visable_mass(dihiggs_10.fold_0_array, 'dihiggs_10')
        ztautau_vis_mass = visable_mass(ztautau.fold_0_array, 'ztautau')
        ttbar_vis_mass = visable_mass(ttbar.fold_0_array, 'ttbar')

        dihiggs_01_target = dihiggs_01_target / dihiggs_01_vis_mass
        dihiggs_10_target = dihiggs_10_target / dihiggs_10_vis_mass
        ztautau_target = ztautau_target / ztautau_vis_mass
        ttbar_target = ttbar_target / ttbar_vis_mass

        features_dihiggs_01 = features_table(dihiggs_01.fold_0_array)
        features_dihiggs_10 = features_table(dihiggs_10.fold_0_array)
        features_ztautau = features_table(ztautau.fold_0_array)
        features_ttbar = features_table(ttbar.fold_0_array)

        len_HH_01 = len(features_dihiggs_01)
        len_HH_10 = len(features_dihiggs_10)
        len_ztautau = len(features_ztautau)
        len_ttbar = len(features_ttbar)
        
        train_features_new = np.concatenate([
            features_dihiggs_01,
            features_dihiggs_10,
            features_ztautau,
            features_ttbar
        ])

        scaler = StandardScaler()
        train_features_new = scaler.fit_transform(X=train_features_new)
        features_dihiggs_01 = train_features_new[:len_HH_01]
        features_dihiggs_10 = train_features_new[len_HH_01:len_HH_01+len_HH_10]
        features_ztautau = train_features_new[len_HH_01+len_HH_10:len_HH_01+len_HH_10+len_ztautau]
        features_ttbar = train_features_new[len_HH_01+len_HH_10+len_ztautau:]

        features_dihiggs_01 = np.append(features_dihiggs_01, [['dihiggs_01']]*len_HH_01, 1)
        features_dihiggs_10 = np.append(features_dihiggs_10, [['dihiggs_10']]*len_HH_10, 1)
        features_ztautau = np.append(features_ztautau, [['ztautau']]*len_ztautau, 1)
        features_ttbar = np.append(features_ttbar, [['ttbar']]*len_ttbar, 1)

        train_target = ak.concatenate([
            dihiggs_01_target,
            dihiggs_10_target,
            ztautau_target,
            ttbar_target
        ])
        train_features = np.concatenate([
            features_dihiggs_01,
            features_dihiggs_10,
            features_ztautau,
            features_ttbar
        ])

        if args.library == 'scikit':
            if args.gridsearch:
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
            regressor = keras_model_main((train_features.shape[1] - 1,))
            _epochs = 25
            _filename = 'cache/my_keras_training.h5'
            X_train, X_test, y_train, y_test = train_test_split(
                train_features, train_target, test_size=0.1, random_state=42)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            # y_train = ak.to_numpy(y_train)
            # y_test = ak.to_numpy(y_test)
            print(type(y_train))
            sample_weights = []
            for i in range(len(X_train)):
                if (X_train[i][-1] == 'ztautau'):
                    sample_weights.append(2.00)
                else:
                    sample_weights.append(1.00)
            sample_weights = np.array(sample_weights)

            X_train_new = []
            for i in range(len(X_train)):
                temp = []
                for j in range(len(X_train[i])):
                    if j != 17:
                        temp.append(float(X_train[i][j]))
                X_train_new.append(temp)

            X_test_new = []
            for i in range(len(X_test)):
                temp = []
                for j in range(len(X_test[i])):
                    if j != 17:
                        temp.append(float(X_test[i][j]))
                X_test_new.append(temp)

            X_train = np.array(X_train_new)
            X_test = np.array(X_test_new)
            
            try:
                rate = 0.003
                batch_size = 1024
                adam = optimizers.get('Adam')
                adam.learning_rate = rate
                regressor.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse', 'mae'])
                history = regressor.fit(
                    X_train, y_train,
                    epochs=_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    sample_weight=sample_weights,
                    ## validation_split=0.1,
                    validation_data=(X_test, y_test),
                    callbacks=[
                        EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
                        ModelCheckpoint(
                            _filename, monitor='val_loss',
                            verbose=True, save_best_only=True)])
                regressor.save(_filename)
                for k in history.history.keys():
                    if 'val' in k:
                        continue
                    nn_history(history, metric=k)

            except KeyboardInterrupt:
                log.info('Ended early..')

        else:
            pass

    log.info('plotting')

    test_target_HH_01  = dihiggs_01.fold_1_array['universal_true_mhh']
    test_target_HH_10  = dihiggs_10.fold_1_array['universal_true_mhh']
    test_target_ztautau =  ztautau.fold_1_array['universal_true_mhh']
    test_target_ttbar  = ttbar.fold_1_array['universal_true_mhh']
        

    features_test_HH_01 = features_table(dihiggs_01.fold_1_array)
    features_test_HH_10 = features_table(dihiggs_10.fold_1_array)
    features_test_ztautau = features_table(ztautau.fold_1_array)
    features_test_ttbar = features_table(ttbar.fold_1_array)
    log.info ('features loaded')

    scaler = StandardScaler()
    len_HH_01 = len(features_test_HH_01)
    len_HH_10 = len(features_test_HH_10)
    len_ztautau = len(features_test_ztautau)
    train_features_new = np.concatenate([
        features_test_HH_01,
        features_test_HH_10,
        features_test_ztautau,
        features_test_ttbar
    ])

    train_features_new = scaler.fit_transform(X=train_features_new)
    log.info ('scaler ran')

    features_test_HH_01 = train_features_new[:len_HH_01]
    features_test_HH_10 = train_features_new[len_HH_01:len_HH_01+len_HH_10]
    features_test_ztautau = train_features_new[len_HH_01+len_HH_10:len_HH_01+len_HH_10+len_ztautau]
    features_test_ttbar = train_features_new[len_HH_01+len_HH_10+len_ztautau:]

    predictions_HH_01 = regressor.predict(features_test_HH_01)
    predictions_HH_10 = regressor.predict(features_test_HH_10)
    predictions_ztautau = regressor.predict(features_test_ztautau)
    predictions_ttbar = regressor.predict(features_test_ttbar)
    log.info ('regressor ran')

    if args.library == 'keras':
        predictions_HH_01 = np.reshape(
            predictions_HH_01, (predictions_HH_01.shape[0], ))
        predictions_HH_10 = np.reshape(
            predictions_HH_10, (predictions_HH_10.shape[0], ))
        predictions_ztautau = np.reshape(
            predictions_ztautau, (predictions_ztautau.shape[0], ))
        predictions_ttbar = np.reshape(
            predictions_ttbar, (predictions_ttbar.shape[0], ))

    mvis_HH_01 = visable_mass(dihiggs_01.fold_1_array, 'dihiggs_01')
    mvis_HH_10 = visable_mass(dihiggs_10.fold_1_array, 'dihiggs_10')
    mvis_ztautau = visable_mass(ztautau.fold_1_array, 'ztautau')
    mvis_ttbar = visable_mass(ttbar.fold_1_array, 'ttbar')
    log.info ('mvis computed')

    predictions_HH_01 = predictions_HH_01 * np.array(mvis_HH_01)
    predictions_HH_10 = predictions_HH_10 * np.array(mvis_HH_10)
    predictions_ztautau = predictions_ztautau * np.array(mvis_ztautau)
    predictions_ttbar = predictions_ttbar * np.array(mvis_ttbar)

    print (dihiggs_01.fold_1_array.fields)
    """
    if 'mmc_bbtautau' in dihiggs_01.fold_1_array.fields:
        mmc_HH_01 = dihiggs_01.fold_1_array['mmc_tautau']
        mhh_mmc_HH_01 = dihiggs_01.fold_1_array['mmc_bbtautau']
    else:
        mmc_HH_01, mhh_mmc_HH_01 = mmc(dihiggs_01.fold_1_array)

    if 'mmc_bbtautau' in dihiggs_10.fold_1_array.fields: 
        mmc_HH_10 = dihiggs_10.fold_1_array['mmc_tautau']
        mhh_mmc_HH_10 = dihiggs_10.fold_1_array['mmc_bbtautau']
    else:
        mmc_HH_10, mhh_mmc_HH_10 = mmc(dihiggs_10.fold_1_array)

    if 'mmc_bbtautau' in ztautau.fold_1_array.fields:
        mmc_ztautau = ztautau.fold_1_array['mmc_tautau']
        mhh_mmc_ztautau = ztautau.fold_1_array['mmc_bbtautau']
    else:
        mmc_ztautau, mhh_mmc_ztautau = mmc(ztautau.fold_1_array)

    if 'mmc_bbtautau' in ttbar.fold_1_array.fields:
        mmc_ttbar = ttbar.fold_1_array['mmc_tautau']
        mhh_mmc_ttbar = ttbar.fold_1_array['mmc_bbtautau']
    else:
        mmc_ttbar, mhh_mmc_ttbar = mmc(ttbar.fold_1_array)
    """
    
    # I know that this is all labeled MMC even though its the original RNN. I'm leaving it like this to avoid changing all of the variable names.
    original_regressor = load_model('cache/original_training.h5')
    mhh_mmc_HH_01 = original_regressor.predict(features_test_HH_01)
    mhh_mmc_HH_10 = original_regressor.predict(features_test_HH_10)
    mhh_mmc_ztautau = original_regressor.predict(features_test_ztautau)
    mhh_mmc_ttbar = original_regressor.predict(features_test_ttbar)
    
    if args.library == 'keras':
        mhh_mmc_HH_01 = np.reshape(
            mhh_mmc_HH_01, (mhh_mmc_HH_01.shape[0], ))
        mhh_mmc_HH_10 = np.reshape(
            mhh_mmc_HH_10, (mhh_mmc_HH_10.shape[0], ))
        mhh_mmc_ztautau = np.reshape(
            mhh_mmc_ztautau, (mhh_mmc_ztautau.shape[0], ))
        mhh_mmc_ttbar = np.reshape(
            mhh_mmc_ttbar, (mhh_mmc_ttbar.shape[0], ))

    mhh_mmc_HH_01 = mhh_mmc_HH_01 * np.array(mvis_HH_01)
    mhh_mmc_HH_10 = mhh_mmc_HH_10 * np.array(mvis_HH_10)
    mhh_mmc_ztautau = mhh_mmc_ztautau * np.array(mvis_ztautau)
    mhh_mmc_ttbar = mhh_mmc_ttbar * np.array(mvis_ttbar)

    eff_HH_01_rnn_mmc, eff_true_HH_01, n_rnn_HH_01, n_mmc_HH_01, n_true_HH_01 = rnn_mmc_comparison(predictions_HH_01, test_target_HH_01, dihiggs_01, dihiggs_01.fold_1_array, 'dihiggs_01', args.library, predictions_mmc = mhh_mmc_HH_01)
    eff_HH_10_rnn_mmc, eff_true_HH_10, n_rnn_HH_10, n_mmc_HH_10, n_true_HH_10 = rnn_mmc_comparison(predictions_HH_10, test_target_HH_10, dihiggs_10, dihiggs_10.fold_1_array, 'dihiggs_10', args.library, predictions_mmc = mhh_mmc_HH_10)
    eff_ztt_rnn_mmc, eff_true_ztt, n_rnn_ztt, n_mmc_ztt, n_true_ztt = rnn_mmc_comparison(predictions_ztautau, test_target_ztautau, ztautau, ztautau.fold_1_array, 'ztautau', args.library, predictions_mmc = mhh_mmc_ztautau)
    eff_ttbar_rnn_mmc, eff_true_ttbar, n_rnn_ttbar, n_mmc_ttbar, n_true_ttbar = rnn_mmc_comparison(predictions_ttbar, test_target_ttbar, ttbar, ttbar.fold_1_array, 'ttbar', args.library, predictions_mmc = mhh_mmc_ttbar)

    # Chi-Square calculations

    # Relevant ROC curves
    eff_pred_HH_01_HH_10 = eff_HH_01_rnn_mmc + eff_HH_10_rnn_mmc
    eff_pred_HH_01_ztt = eff_HH_01_rnn_mmc + eff_ztt_rnn_mmc
    eff_pred_HH_01_ttbar = eff_HH_01_rnn_mmc + eff_ttbar_rnn_mmc
    eff_true_HH_01_HH_10 = [eff_true_HH_01] + [eff_true_HH_10]
    eff_true_HH_01_ztt = [eff_true_HH_01] + [eff_true_ztt]
    eff_true_HH_01_ttbar = [eff_true_HH_01] + [eff_true_ttbar]
    roc_plot_rnn_mmc(eff_pred_HH_01_HH_10, eff_true_HH_01_HH_10, r'$\kappa_{\lambda}$ = 1', r'$\kappa_{\lambda}$ = 10')
    roc_plot_rnn_mmc(eff_pred_HH_01_ztt, eff_true_HH_01_ztt, r'$\kappa_{\lambda}$ = 1', r'$Z\to\tau\tau$ + jets')
    roc_plot_rnn_mmc(eff_pred_HH_01_ttbar, eff_true_HH_01_ttbar, r'$\kappa_{\lambda}$ = 1', 'Top Quark')

    # Pile-up stability of the signal
    avg_mhh_HH_01 = avg_mhh_calculation(dihiggs_01.fold_1_array, test_target_HH_01, predictions_HH_01, mhh_mmc_HH_01)
    avg_mhh_HH_10 = avg_mhh_calculation(dihiggs_10.fold_1_array, test_target_HH_10, predictions_HH_10, mhh_mmc_HH_10)
    avg_mhh_plot(avg_mhh_HH_01, 'pileup_stability_avg_mhh_HH_01', dihiggs_01)
    avg_mhh_plot(avg_mhh_HH_10, 'pileup_stability_avg_mhh_HH_10', dihiggs_10)
