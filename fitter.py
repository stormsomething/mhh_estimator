import os
import joblib
import awkward as ak
import numpy as np
import scipy.stats
import tensorflow as tf
import uproot
from argparse import ArgumentParser
from bbtautau import log; log = log.getChild('fitter')
from bbtautau.utils import features_table, universal_true_mhh, visable_mass, clean_samples, chi_square_test, rotate_events
from bbtautau.plotting import signal_features, ztautau_pred_target_comparison, roc_plot_rnn_mmc, rnn_mmc_comparison, avg_mhh_calculation, avg_mhh_plot
from bbtautau.database import dihiggs_01, dihiggs_10, ztautau, ttbar
from bbtautau.models import keras_model_main
from bbtautau.plotting import nn_history, sigma_plots, resid_comparison_plots, k_lambda_comparison_plot, reweight_plot, resolution_plot, klambda_scan_plot, reweight_and_compare
from bbtautau.mmc import mmc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras import backend

def tf_mdn_loss(y, model, sample_weight=None):
    if sample_weight is not None:
        return -model.log_prob(y) * sample_weight
    return -model.log_prob(y)

def gaussian_nll(y_true, y_pred, sample_weight=None):
    # From https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e
    mu = y_pred[:,0]
    logsigma = y_pred[:,1]
    
    mse = -0.5*backend.square((y_true-mu)/backend.exp(logsigma))
    log2pi = -0.5*np.log(2*np.pi)
    
    if sample_weight is not None:
        print('Using Sample Weight!')
        log_likelihood = (mse - logsigma + log2pi) * sample_weight
        return -backend.sum(log_likelihood) / backend.sum(sample_weight)
        
    print('NOT Using Sample Weight!')
    log_likelihood = mse - logsigma + log2pi
    return -backend.mean(log_likelihood)

def mse_of_mu(y_true, y_pred):
    mu = y_pred[:,0]
    return tf.keras.losses.mean_squared_error(y_true, mu)
    
def sharp_peak_loss(y_true, y_pred):
    # loss function that tries to force mu=1000 and sigma=100
    mu = y_pred[:,0]
    logsigma = y_pred[:,1]
    
    mu_sq_err = backend.square(mu - 1000)
    sigma_sq_err = backend.square(backend.exp(logsigma) - 100)
    
    return backend.sum(mu_sq_err + sigma_sq_err)

def gaussian_nll_np(y_true, mu, sigma, sample_weight=None):
    mse = -0.5*np.square((y_true-mu)/sigma)
    log2pi = -0.5*np.log(2*np.pi)
    
    if sample_weight is not None:
        log_likelihood = (mse - np.log(sigma) + log2pi) * sample_weight
        return np.sum(-log_likelihood) / np.sum(sample_weight)
        
    log_likelihood = mse - np.log(sigma) + log2pi
    return np.mean(-log_likelihood)

def mse_of_mu_np(y_true, mu, sample_weight=None):
    sq_err = np.square(y_true-mu)
    
    if sample_weight is not None:
        weighted_sq_err = sq_err * sample_weight
        return np.sum(weighted_sq_err) / np.sum(sample_weight)
        
    return np.mean(sq_err)

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
            regressor = load_model('cache/my_keras_training.h5', custom_objects={'tf_mdn_loss': tf_mdn_loss})
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
            print("Columns in Training Features:")
            print(train_features.shape[1])
            regressor = keras_model_main((train_features.shape[1] - 1,))
            _epochs = 200
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
                    if j != train_features.shape[1] - 1:
                        temp.append(float(X_train[i][j]))
                X_train_new.append(temp)

            X_test_new = []
            for i in range(len(X_test)):
                temp = []
                for j in range(len(X_test[i])):
                    if j != train_features.shape[1] - 1:
                        temp.append(float(X_test[i][j]))
                X_test_new.append(temp)

            X_train = np.array(X_train_new)
            X_test = np.array(X_test_new)
            
            try:
                rate = 2e-6 # default 0.001
                batch_size = 64
                adam = optimizers.get('Adam')
                #adam = optimizers.get('Nadam')
                #adam = optimizers.get('SGD')
                adam.learning_rate = rate
                adam.beta_1 = 0.9 # default 0.9
                adam.beta_2 = 0.999 # default 0.999
                adam.epsilon = 1e-7 # default 1e-7
                """
                adam.momentum = 0.9
                adam.nesterov = True
                """
                #regressor = load_model('cache/my_keras_training.h5', custom_objects={'tf_mdn_loss': tf_mdn_loss})
                # For use with Tensorflow MixtureNormal
                regressor.compile(loss=tf_mdn_loss, optimizer=adam)
                # For use with "fake" single Gaussian MDN that's actually just a 2-output NN (mu, logsigma)
                #regressor.compile(loss=gaussian_nll, optimizer=adam)
                #regressor.compile(loss=gaussian_nll, optimizer=adam, metrics=['mse', 'mae'])
                #regressor.compile(loss=sharp_peak_loss, optimizer=adam, metrics=['mse', 'mae'])
                history = regressor.fit(
                    X_train, y_train,
                    epochs=_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    sample_weight=sample_weights,
                    ## validation_split=0.1,
                    validation_data=(X_test, y_test),
                    callbacks=[
                        EarlyStopping(verbose=True, patience=10, monitor='val_loss'),
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
    
    """
    print(features_test_HH_01.shape)
    for i in range(features_test_HH_01.shape[1]):
        print(scipy.stats.describe(features_test_HH_01[:,i]))
    """
    
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

    model_HH_01 = regressor(features_test_HH_01)
    model_HH_10 = regressor(features_test_HH_10)
    model_ztautau = regressor(features_test_ztautau)
    model_ttbar = regressor(features_test_ttbar)
    
    predictions_HH_01 = np.array(model_HH_01.mean())
    predictions_HH_10 = np.array(model_HH_10.mean())
    predictions_ztautau = np.array(model_ztautau.mean())
    predictions_ttbar = np.array(model_ttbar.mean())
    
    sigmas_HH_01 = np.array(model_HH_01.stddev())
    sigmas_HH_10 = np.array(model_HH_10.stddev())
    sigmas_ztautau = np.array(model_ztautau.stddev())
    sigmas_ttbar = np.array(model_ttbar.stddev())
    
    log.info ('regressor ran')

    if args.library == 'keras':
        sigmas_HH_01 = np.reshape(
            sigmas_HH_01, (sigmas_HH_01.shape[0], ))
        sigmas_HH_10 = np.reshape(
            sigmas_HH_10, (sigmas_HH_10.shape[0], ))
        sigmas_ztautau = np.reshape(
            sigmas_ztautau, (sigmas_ztautau.shape[0], ))
        sigmas_ttbar = np.reshape(
            sigmas_ttbar, (sigmas_ttbar.shape[0], ))
            
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

    predictions_HH_01 *= np.array(mvis_HH_01)
    predictions_HH_10 *= np.array(mvis_HH_10)
    predictions_ztautau *= np.array(mvis_ztautau)
    predictions_ttbar *= np.array(mvis_ttbar)
    
    sigmas_HH_01 *= np.array(mvis_HH_01)
    sigmas_HH_10 *= np.array(mvis_HH_10)
    sigmas_ztautau *= np.array(mvis_ztautau)
    sigmas_ttbar *= np.array(mvis_ttbar)
    
    """
    print('The losses (calculated outside of Keras, after normalization by visible mass) are:')
    print('dihiggs_01: ' + str(gaussian_nll_np(test_target_HH_01, predictions_HH_01, sigmas_HH_01)))
    print('dihiggs_10: ' + str(gaussian_nll_np(test_target_HH_10, predictions_HH_10, sigmas_HH_10)))
    print('ztautau: ' + str(gaussian_nll_np(test_target_ztautau, predictions_ztautau, sigmas_ztautau)))
    print('ttbar: ' + str(gaussian_nll_np(test_target_ttbar, predictions_ttbar, sigmas_ttbar)))
    """

    print (dihiggs_01.fold_1_array.fields)
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
        
    print("Stats for 4 samples where m_tautau is 0:")
    print(scipy.stats.describe(mmc_HH_01[np.where(mmc_HH_01 == 0)]))
    print(scipy.stats.describe(mmc_HH_10[np.where(mmc_HH_10 == 0)]))
    print(scipy.stats.describe(mmc_ztautau[np.where(mmc_ztautau == 0)]))
    print(scipy.stats.describe(mmc_ttbar[np.where(mmc_ttbar == 0)]))
    print("Stats for 4 samples where m_tautau is not 0:")
    print(scipy.stats.describe(mmc_HH_01[np.where(mmc_HH_01 > 0)]))
    print(scipy.stats.describe(mmc_HH_10[np.where(mmc_HH_10 > 0)]))
    print(scipy.stats.describe(mmc_ztautau[np.where(mmc_ztautau > 0)]))
    print(scipy.stats.describe(mmc_ttbar[np.where(mmc_ttbar > 0)]))
    """
    print("Info about Cache Contents:")
    for field in dihiggs_01.fold_1_array.fields:
        print(str(field))
        print(type(dihiggs_01.fold_1_array[field][0]))
    """
    
    # I know that this is all labeled MMC even though its the original RNN. I'm leaving it like this to avoid changing all of the variable names.
    log.info ('Loading Old Model for Comparison')
    """
    original_regressor = load_model('cache/original_training.h5')
    mhh_original_HH_01 = original_regressor.predict(features_test_HH_01)
    mhh_original_HH_10 = original_regressor.predict(features_test_HH_10)
    mhh_original_ztautau = original_regressor.predict(features_test_ztautau)
    mhh_original_ttbar = original_regressor.predict(features_test_ttbar)
    
    
    if args.library == 'keras':
        mhh_original_HH_01 = np.reshape(
            mhh_original_HH_01, (mhh_original_HH_01.shape[0], ))
        mhh_original_HH_10 = np.reshape(
            mhh_original_HH_10, (mhh_original_HH_10.shape[0], ))
        mhh_original_ztautau = np.reshape(
            mhh_original_ztautau, (mhh_original_ztautau.shape[0], ))
        mhh_original_ttbar = np.reshape(
            mhh_original_ttbar, (mhh_original_ttbar.shape[0], ))
            
    mhh_original_HH_01 *= np.array(mvis_HH_01)
    mhh_original_HH_10 *= np.array(mvis_HH_10)
    mhh_original_ztautau *= np.array(mvis_ztautau)
    mhh_original_ttbar *= np.array(mvis_ttbar)
    """
    
    # Use this to replace original regressor with a copy of MDN (use when the input variable setup is mismatched)
    mhh_original_HH_01 = predictions_HH_01
    mhh_original_HH_10 = predictions_HH_10
    mhh_original_ztautau = predictions_ztautau
    mhh_original_ttbar = predictions_ttbar
    
    print('The number of events in each sample are:')
    print('dihiggs_01: ' + str(len(predictions_HH_01)))
    print('dihiggs_10: ' + str(len(predictions_HH_10)))
    print('ztautau: ' + str(len(predictions_ztautau)))
    print('ttbar: ' + str(len(predictions_ttbar)))
    print('The number of MMC-failed events in each sample are:')
    print('dihiggs_01: ' + str(len(predictions_HH_01[np.where(mhh_mmc_HH_01 < 200)])))
    print('dihiggs_10: ' + str(len(predictions_HH_10[np.where(mhh_mmc_HH_10 < 200)])))
    
    all_predictions = np.concatenate([
        predictions_HH_01,
        predictions_HH_10,
        predictions_ztautau,
        predictions_ttbar
    ])
    all_sigmas = np.concatenate([
        sigmas_HH_01,
        sigmas_HH_10,
        sigmas_ztautau,
        sigmas_ttbar
    ])
    all_fold_1_arrays = np.concatenate([
        dihiggs_01.fold_1_array,
        dihiggs_10.fold_1_array,
        ztautau.fold_1_array,
        ttbar.fold_1_array
    ])
    all_original = np.concatenate([
        mhh_original_HH_01,
        mhh_original_HH_10,
        mhh_original_ztautau,
        mhh_original_ttbar
    ])
    all_mmc = np.concatenate([
        mhh_mmc_HH_01,
        mhh_mmc_HH_10,
        mhh_mmc_ztautau,
        mhh_mmc_ttbar
    ])
    all_mvis = np.concatenate([
        mvis_HH_01,
        mvis_HH_10,
        mvis_ztautau,
        mvis_ttbar
    ])
    
    resol_HH_01 = resolution_plot(predictions_HH_01, sigmas_HH_01, dihiggs_01.fold_1_array, 'dihiggs_01_mdn')
    resol_HH_10 = resolution_plot(predictions_HH_10, sigmas_HH_10, dihiggs_10.fold_1_array, 'dihiggs_10_mdn')
    resol_ztautau = resolution_plot(predictions_ztautau, sigmas_ztautau, ztautau.fold_1_array, 'ztautau_mdn')
    resol_ttbar = resolution_plot(predictions_ttbar, sigmas_ttbar, ttbar.fold_1_array, 'ttbar_mdn')
    resolution_plot(all_predictions, all_sigmas, all_fold_1_arrays, 'all_mdn')
    resolution_plot(np.array(dihiggs_01.fold_1_array['universal_true_mhh']), sigmas_HH_01, dihiggs_01.fold_1_array, 'dihiggs_01_truth')
    resolution_plot(np.array(dihiggs_10.fold_1_array['universal_true_mhh']), sigmas_HH_10, dihiggs_10.fold_1_array, 'dihiggs_10_truth')
    resolution_plot(np.array(ztautau.fold_1_array['universal_true_mhh']), sigmas_ztautau, ztautau.fold_1_array, 'ztautau_truth')
    resolution_plot(np.array(ttbar.fold_1_array['universal_true_mhh']), sigmas_ttbar, ttbar.fold_1_array, 'ttbar_truth')
    resolution_plot(np.array(all_fold_1_arrays['universal_true_mhh']), all_sigmas, all_fold_1_arrays, 'all_truth')
    
    # Use this to split by resolution rather than sigma
    sigmas_HH_01 /= resol_HH_01
    sigmas_HH_10 /= resol_HH_10
    sigmas_ztautau /= resol_ztautau
    sigmas_ttbar /= resol_ttbar
    all_sigmas = np.concatenate([
        sigmas_HH_01,
        sigmas_HH_10,
        sigmas_ztautau,
        sigmas_ttbar
    ])
    indices_1_1 = np.where(sigmas_HH_01 < 4)
    indices_1_10 = np.where(sigmas_HH_10 < 4)
    indices_1_z = np.where(sigmas_ztautau < 4)
    indices_1_t = np.where(sigmas_ttbar < 4)
    indices_2_1 = np.where(sigmas_HH_01 > 4)
    indices_2_10 = np.where(sigmas_HH_10 > 4)
    indices_2_z = np.where(sigmas_ztautau > 4)
    indices_2_t = np.where(sigmas_ttbar > 4)
    
    """
    log.info ('Beginning Sigma Plotting')
    
    sigma_plots(predictions_HH_01, sigmas_HH_01, dihiggs_01.fold_1_array, 'dihiggs_01', mvis_HH_01)
    sigma_plots(predictions_HH_10, sigmas_HH_10, dihiggs_10.fold_1_array, 'dihiggs_10', mvis_HH_10)
    sigma_plots(predictions_ztautau, sigmas_ztautau, ztautau.fold_1_array, 'ztautau', mvis_ztautau)
    sigma_plots(predictions_ttbar, sigmas_ttbar, ttbar.fold_1_array, 'ttbar', mvis_ttbar)
    sigma_plots(all_predictions, all_sigmas, all_fold_1_arrays, 'all', all_mvis)
    
    resid_comparison_plots(predictions_HH_01, sigmas_HH_01, mhh_original_HH_01, mhh_mmc_HH_01, dihiggs_01.fold_1_array, 'dihiggs_01', mvis_HH_01)
    resid_comparison_plots(predictions_HH_10, sigmas_HH_10, mhh_original_HH_10, mhh_mmc_HH_10, dihiggs_10.fold_1_array, 'dihiggs_10', mvis_HH_10)
    resid_comparison_plots(predictions_ztautau, sigmas_ztautau, mhh_original_ztautau, mhh_mmc_ztautau, ztautau.fold_1_array, 'ztautau', mvis_ztautau)
    resid_comparison_plots(predictions_ttbar, sigmas_ttbar, mhh_original_ttbar, mhh_mmc_ttbar, ttbar.fold_1_array, 'ttbar', mvis_ttbar)
    resid_comparison_plots(all_predictions, all_sigmas, all_original, all_mmc, all_fold_1_arrays, 'all', all_mvis)

    
    log.info ('Finished Sigma Plotting, Beginning k_lambda Comparison Plotting')
    
    k_lambda_comparison_plot(dihiggs_01.fold_1_array['universal_true_mhh'], dihiggs_10.fold_1_array['universal_true_mhh'], dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, 'truth')
    k_lambda_comparison_plot(predictions_HH_01, predictions_HH_10, dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, 'mdn')
    k_lambda_comparison_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, 'mmc')
    k_lambda_comparison_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, 'mmc_fail', slice_indices_1 = np.where(mmc_HH_01 == 0), slice_indices_10 = np.where(mmc_HH_10 == 0))
    k_lambda_comparison_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, 'mmc_pass', slice_indices_1 = np.where(mmc_HH_01 > 0), slice_indices_10 = np.where(mmc_HH_10 > 0))
    k_lambda_comparison_plot(mhh_original_HH_01, mhh_original_HH_10, dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, 'dnn')
    
    k_lambda_comparison_plot(predictions_HH_01[indices_1_1], predictions_HH_10[indices_1_10], dihiggs_01.fold_1_array[indices_1_1], dihiggs_10.fold_1_array[indices_1_10], 'mdn_low_sigma')
    k_lambda_comparison_plot(predictions_HH_01[indices_2_1], predictions_HH_10[indices_2_10], dihiggs_01.fold_1_array[indices_2_1], dihiggs_10.fold_1_array[indices_2_10], 'mdn_high_sigma')
   
    log.info ('Finished k_lambda Comparison Plotting, Beginning RNN-MMC Comparison Plotting')
    
    eff_HH_01_rnn_mmc, eff_true_HH_01, n_rnn_HH_01, n_mmc_HH_01, n_true_HH_01 = rnn_mmc_comparison(predictions_HH_01, test_target_HH_01, dihiggs_01, dihiggs_01.fold_1_array, 'dihiggs_01', args.library, predictions_old = mhh_original_HH_01, predictions_mmc = mhh_mmc_HH_01)
    eff_HH_10_rnn_mmc, eff_true_HH_10, n_rnn_HH_10, n_mmc_HH_10, n_true_HH_10 = rnn_mmc_comparison(predictions_HH_10, test_target_HH_10, dihiggs_10, dihiggs_10.fold_1_array, 'dihiggs_10', args.library, predictions_old = mhh_original_HH_10, predictions_mmc = mhh_mmc_HH_10)
    eff_ztt_rnn_mmc, eff_true_ztt, n_rnn_ztt, n_mmc_ztt, n_true_ztt = rnn_mmc_comparison(predictions_ztautau, test_target_ztautau, ztautau, ztautau.fold_1_array, 'ztautau', args.library, predictions_old = mhh_original_ztautau, predictions_mmc = mhh_mmc_ztautau)
    eff_ttbar_rnn_mmc, eff_true_ttbar, n_rnn_ttbar, n_mmc_ttbar, n_true_ttbar = rnn_mmc_comparison(predictions_ttbar, test_target_ttbar, ttbar, ttbar.fold_1_array, 'ttbar', args.library, predictions_old = mhh_original_ttbar, predictions_mmc = mhh_mmc_ttbar)

    log.info ('Finished RNN-MMC Comparison Plotting')

    # Chi-Square calculations

    # Relevant ROC curves
    eff_pred_HH_01_HH_10 = eff_HH_01_rnn_mmc + eff_HH_10_rnn_mmc
    eff_pred_HH_01_ztt = eff_HH_01_rnn_mmc + eff_ztt_rnn_mmc
    eff_pred_HH_01_ttbar = eff_HH_01_rnn_mmc + eff_ttbar_rnn_mmc
    eff_true_HH_01_HH_10 = [eff_true_HH_01] + [eff_true_HH_10]
    eff_true_HH_01_ztt = [eff_true_HH_01] + [eff_true_ztt]
    eff_true_HH_01_ttbar = [eff_true_HH_01] + [eff_true_ttbar]
    print("ROC Array Shapes:")
    print(len(eff_pred_HH_01_HH_10[0]))
    print(len(eff_pred_HH_01_HH_10[1]))
    print(len(eff_pred_HH_01_HH_10[2]))
    print(len(eff_pred_HH_01_HH_10[3]))
    print(len(eff_true_HH_01_HH_10[0]))
    print(len(eff_true_HH_01_HH_10[1]))
    roc_plot_rnn_mmc(eff_pred_HH_01_HH_10, eff_true_HH_01_HH_10, r'$\kappa_{\lambda}$ = 1', r'$\kappa_{\lambda}$ = 10')
    roc_plot_rnn_mmc(eff_pred_HH_01_ztt, eff_true_HH_01_ztt, r'$\kappa_{\lambda}$ = 1', r'$Z\to\tau\tau$ + jets')
    roc_plot_rnn_mmc(eff_pred_HH_01_ttbar, eff_true_HH_01_ttbar, r'$\kappa_{\lambda}$ = 1', 'Top Quark')
    
    log.info ('Beginning Sigma-Split RNN-MMC Comparison Plotting')
    
    # lowest sigma
    eff_HH_01_rnn_mmc, eff_true_HH_01, n_rnn_HH_01, n_mmc_HH_01, n_true_HH_01 = rnn_mmc_comparison(predictions_HH_01, test_target_HH_01, dihiggs_01, dihiggs_01.fold_1_array, 'dihiggs_01', args.library, predictions_old = mhh_original_HH_01, predictions_mmc = mhh_mmc_HH_01, sigma_label = '_low_sigma', sigma_slice = indices_1_1)
    eff_HH_10_rnn_mmc, eff_true_HH_10, n_rnn_HH_10, n_mmc_HH_10, n_true_HH_10 = rnn_mmc_comparison(predictions_HH_10, test_target_HH_10, dihiggs_10, dihiggs_10.fold_1_array, 'dihiggs_10', args.library, predictions_old = mhh_original_HH_10, predictions_mmc = mhh_mmc_HH_10, sigma_label = '_low_sigma', sigma_slice = indices_1_10)
    eff_ztt_rnn_mmc, eff_true_ztt, n_rnn_ztt, n_mmc_ztt, n_true_ztt = rnn_mmc_comparison(predictions_ztautau, test_target_ztautau, ztautau, ztautau.fold_1_array, 'ztautau', args.library, predictions_old = mhh_original_ztautau, predictions_mmc = mhh_mmc_ztautau, sigma_label = '_low_sigma', sigma_slice = indices_1_z)
    eff_ttbar_rnn_mmc, eff_true_ttbar, n_rnn_ttbar, n_mmc_ttbar, n_true_ttbar = rnn_mmc_comparison(predictions_ttbar, test_target_ttbar, ttbar, ttbar.fold_1_array, 'ttbar', args.library, predictions_old = mhh_original_ttbar, predictions_mmc = mhh_mmc_ttbar, sigma_label = '_low_sigma', sigma_slice = indices_1_t)
    eff_pred_HH_01_HH_10 = eff_HH_01_rnn_mmc + eff_HH_10_rnn_mmc
    eff_pred_HH_01_ztt = eff_HH_01_rnn_mmc + eff_ztt_rnn_mmc
    eff_pred_HH_01_ttbar = eff_HH_01_rnn_mmc + eff_ttbar_rnn_mmc
    eff_true_HH_01_HH_10 = [eff_true_HH_01] + [eff_true_HH_10]
    eff_true_HH_01_ztt = [eff_true_HH_01] + [eff_true_ztt]
    eff_true_HH_01_ttbar = [eff_true_HH_01] + [eff_true_ttbar]
    roc_plot_rnn_mmc(eff_pred_HH_01_HH_10, eff_true_HH_01_HH_10, r'$\kappa_{\lambda}$ = 1', r'$\kappa_{\lambda}$ = 10', sigma_label = '_low_sigma')
    roc_plot_rnn_mmc(eff_pred_HH_01_ztt, eff_true_HH_01_ztt, r'$\kappa_{\lambda}$ = 1', r'$Z\to\tau\tau$ + jets', sigma_label = '_low_sigma')
    roc_plot_rnn_mmc(eff_pred_HH_01_ttbar, eff_true_HH_01_ttbar, r'$\kappa_{\lambda}$ = 1', 'Top Quark', sigma_label = '_low_sigma')
    
    # highest sigma
    eff_HH_01_rnn_mmc, eff_true_HH_01, n_rnn_HH_01, n_mmc_HH_01, n_true_HH_01 = rnn_mmc_comparison(predictions_HH_01, test_target_HH_01, dihiggs_01, dihiggs_01.fold_1_array, 'dihiggs_01', args.library, predictions_old = mhh_original_HH_01, predictions_mmc = mhh_mmc_HH_01, sigma_label = '_high_sigma', sigma_slice = indices_2_1)
    eff_HH_10_rnn_mmc, eff_true_HH_10, n_rnn_HH_10, n_mmc_HH_10, n_true_HH_10 = rnn_mmc_comparison(predictions_HH_10, test_target_HH_10, dihiggs_10, dihiggs_10.fold_1_array, 'dihiggs_10', args.library, predictions_old = mhh_original_HH_10, predictions_mmc = mhh_mmc_HH_10, sigma_label = '_high_sigma', sigma_slice = indices_2_10)
    eff_ztt_rnn_mmc, eff_true_ztt, n_rnn_ztt, n_mmc_ztt, n_true_ztt = rnn_mmc_comparison(predictions_ztautau, test_target_ztautau, ztautau, ztautau.fold_1_array, 'ztautau', args.library, predictions_old = mhh_original_ztautau, predictions_mmc = mhh_mmc_ztautau, sigma_label = '_high_sigma', sigma_slice = indices_2_z)
    eff_ttbar_rnn_mmc, eff_true_ttbar, n_rnn_ttbar, n_mmc_ttbar, n_true_ttbar = rnn_mmc_comparison(predictions_ttbar, test_target_ttbar, ttbar, ttbar.fold_1_array, 'ttbar', args.library, predictions_old = mhh_original_ttbar, predictions_mmc = mhh_mmc_ttbar, sigma_label = '_high_sigma', sigma_slice = indices_2_t)
    eff_pred_HH_01_HH_10 = eff_HH_01_rnn_mmc + eff_HH_10_rnn_mmc
    eff_pred_HH_01_ztt = eff_HH_01_rnn_mmc + eff_ztt_rnn_mmc
    eff_pred_HH_01_ttbar = eff_HH_01_rnn_mmc + eff_ttbar_rnn_mmc
    eff_true_HH_01_HH_10 = [eff_true_HH_01] + [eff_true_HH_10]
    eff_true_HH_01_ztt = [eff_true_HH_01] + [eff_true_ztt]
    eff_true_HH_01_ttbar = [eff_true_HH_01] + [eff_true_ttbar]
    roc_plot_rnn_mmc(eff_pred_HH_01_HH_10, eff_true_HH_01_HH_10, r'$\kappa_{\lambda}$ = 1', r'$\kappa_{\lambda}$ = 10', sigma_label = '_high_sigma')
    roc_plot_rnn_mmc(eff_pred_HH_01_ztt, eff_true_HH_01_ztt, r'$\kappa_{\lambda}$ = 1', r'$Z\to\tau\tau$ + jets', sigma_label = '_high_sigma')
    roc_plot_rnn_mmc(eff_pred_HH_01_ttbar, eff_true_HH_01_ttbar, r'$\kappa_{\lambda}$ = 1', 'Top Quark', sigma_label = '_high_sigma')

    # Pile-up stability of the signal
    avg_mhh_HH_01 = avg_mhh_calculation(dihiggs_01.fold_1_array, test_target_HH_01, predictions_HH_01, mhh_mmc_HH_01)
    avg_mhh_HH_10 = avg_mhh_calculation(dihiggs_10.fold_1_array, test_target_HH_10, predictions_HH_10, mhh_mmc_HH_10)
    avg_mhh_plot(avg_mhh_HH_01, 'pileup_stability_avg_mhh_HH_01', dihiggs_01)
    avg_mhh_plot(avg_mhh_HH_10, 'pileup_stability_avg_mhh_HH_10', dihiggs_10)
    
    # Attempt to reweight klambda=1 to klambda=2
    log.info('Loading Reweight Root Files')
    reweight_file_1 = uproot.open("data/weight-mHH-from-cHHHp01d0-to-cHHHpx_20GeV_Jul28.root")
    reweight_1 = reweight_file_1["reweight_mHH_1p0_to_10p0"].to_numpy()
    norm = reweight_file_1["norm10p0"].value
    reweight_file_10 = uproot.open("data/weight-mHH-from-cHHHp10d0-to-cHHHpx_20GeV_Jul28.root")
    reweight_10 = reweight_file_10["reweight_mHH_1p0_to_1p0"].to_numpy()
    log.info('Beginning Reweight Plots')
    reweight_plot(dihiggs_01.fold_1_array['universal_true_mhh'], dihiggs_10.fold_1_array['universal_true_mhh'], dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, reweight_1, reweight_10, norm, 'truth')
    reweight_plot(predictions_HH_01, predictions_HH_10, dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, reweight_1, reweight_10, norm, 'mdn')
    reweight_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_1_array, dihiggs_10.fold_1_array, reweight_1, reweight_10, norm, 'mmc')
    """
    
    # Scan over a range of klambda values and find their significances
    klambda_scan_list = ['n8p0', 'n7p0', 'n6p0', 'n5p0', 'n4p0', 'n3p0', 'n2p0', 'n1p0', '0p0', '1p0', '2p0', '3p0', '4p0', '5p0', '6p0', '7p0', '8p0', '9p0', '10p0', '11p0']
    reweight_file = uproot.open("data/weight-mHH-from-cHHHp01d0-to-cHHHpx_20GeV_Jul28.root")
    
    truth_significances = []
    mdn_significances = []
    mmc_significances = []
    original_weights = dihiggs_01.fold_1_array['EventInfo___NominalAuxDyn']['evtweight'] * dihiggs_01.fold_1_array['fold_weight']
    
    for klambda in klambda_scan_list:
    
        print('klambda reweight scan: ' + klambda)
        
        if (klambda == '1p0'):
            truth_significances.append(0)
            mdn_significances.append(0)
            mmc_significances.append(0)
            continue
            
        reweight = reweight_file['reweight_mHH_1p0_to_' + klambda].to_numpy()
        norm = reweight_file['norm' + klambda].value
        
        reweights_by_bin = reweight[0] * norm
        num_bins = len(reweights_by_bin)
        
        new_weights = []
        for i in range(len(dihiggs_01.fold_1_array['universal_true_mhh'])):
            reweight_bin = int((dihiggs_01.fold_1_array['universal_true_mhh'][i] - 200) / 20)
            if ((reweight_bin > -1) and (reweight_bin < num_bins)):
                new_weights.append(original_weights[i] * reweights_by_bin[reweight_bin])
            else:
                new_weights.append(0)
                
        z = reweight_and_compare(dihiggs_01.fold_1_array['universal_true_mhh'], original_weights, new_weights, 'truth', klambda)
        truth_significances.append(z)
        
        z = reweight_and_compare(predictions_HH_01, original_weights, new_weights, 'mdn', klambda)
        mdn_significances.append(z)
        
        z = reweight_and_compare(mhh_mmc_HH_01, original_weights, new_weights, 'mmc', klambda)
        mmc_significances.append(z)
        
    klambda_scan_plot(range(-8, 12), truth_significances, mdn_significances, mmc_significances)
        
        