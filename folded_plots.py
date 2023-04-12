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
from bbtautau.plotting import nn_history, sigma_plots, resid_comparison_plots, k_lambda_comparison_plot, reweight_plot, resolution_plot, klambda_scan_plot, reweight_and_compare, eta_plot, res_plots, simple_sigma_plot, separation_overlay_plot, k_lambda_comparison_reader_overlay
from bbtautau.mmc import mmc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.utils.vis_utils import plot_model
from keras import backend

def gaussian_loss(targ, pred, sample_weight=None):
    """
    Basic gaussian loss model. Probably not properly normalized
    From: https://gitlab.cern.ch/atlas-flavor-tagging-tools/trigger/dipz/-/blob/main/keras/train.py#L186-L203
    """
    z = pred[:,0:1]
    q = pred[:,1:2]
    loss = -q + backend.square(z - targ) * backend.exp(q)
    if sample_weight is not None:
        return loss * sample_weight
    return loss
    
def gaussian_loss_prec(targ, pred, sample_weight=None):
    """
    This seems to be more stable than the gaussian loss above
    """
    z = pred[:,0:1]
    prec = backend.abs(pred[:,1:2]) + 1e-6
    loss = - backend.log(prec) + backend.square(z - targ) * prec
    if sample_weight is not None:
        return loss * sample_weight
    return loss

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
    parser.add_argument('--library', default='scikit', choices=['scikit', 'keras'])
    parser.add_argument('--use-cache', default=False, action='store_true')
    args = parser.parse_args()

    # use all root files by default
    max_files = None

    log.info('loading samples ..')

    modulus_options = (3,3) # modulus (fraction) for train-test split, rotation number for train-test split (starts at 0)
    dihiggs_01.process(
        verbose=True,
        max_files=max_files,
        use_cache=args.use_cache,
        remove_bad_training_events=True,
        modulus_options=modulus_options)
    dihiggs_10.process(
        verbose=True,
        max_files=max_files,
        use_cache=args.use_cache,
        remove_bad_training_events=True,
        modulus_options=modulus_options)
    ztautau.process(
        verbose=True,
        is_signal=False,
        max_files=max_files,
        use_cache=args.use_cache,
        remove_bad_training_events=True,
        modulus_options=modulus_options)
    ttbar.process(
        verbose=True,
        is_signal=False,
        max_files=max_files,
        use_cache=args.use_cache,
        remove_bad_training_events=True,
        modulus_options=modulus_options)
        
    log.info('Fold 0 Events HH01: ' + str(len(dihiggs_01.fold_0_array['universal_true_mhh'])))
    log.info('Fold 0 Events HH10: ' + str(len(dihiggs_10.fold_0_array['universal_true_mhh'])))
    log.info('Fold 0 Events ztautau: ' + str(len(ztautau.fold_0_array['universal_true_mhh'])))
    log.info('Fold 0 Events ttbar: ' + str(len(ttbar.fold_0_array['universal_true_mhh'])))
    log.info('Fold 1 Events HH01: ' + str(len(dihiggs_01.fold_1_array['universal_true_mhh'])))
    log.info('Fold 1 Events HH10: ' + str(len(dihiggs_10.fold_1_array['universal_true_mhh'])))
    log.info('Fold 1 Events ztautau: ' + str(len(ztautau.fold_1_array['universal_true_mhh'])))
    log.info('Fold 1 Events ttbar: ' + str(len(ttbar.fold_1_array['universal_true_mhh'])))
        
    log.info('..done')

    log.info('loading regressor weights')
    regressor_0 = load_model('cache/training-0.h5', custom_objects={'gaussian_loss': gaussian_loss, 'gaussian_loss_prec': gaussian_loss_prec})
    regressor_1 = load_model('cache/training-1.h5', custom_objects={'gaussian_loss': gaussian_loss, 'gaussian_loss_prec': gaussian_loss_prec})
    regressor_2 = load_model('cache/training-2.h5', custom_objects={'gaussian_loss': gaussian_loss, 'gaussian_loss_prec': gaussian_loss_prec})

    log.info('plotting')

    test_target_HH_01  = dihiggs_01.fold_0_array['universal_true_mhh']
    test_target_HH_10  = dihiggs_10.fold_0_array['universal_true_mhh']
    test_target_ztautau =  ztautau.fold_0_array['universal_true_mhh']
    test_target_ttbar  = ttbar.fold_0_array['universal_true_mhh']

    features_test_HH_01 = features_table(dihiggs_01.fold_0_array)
    features_test_HH_10 = features_table(dihiggs_10.fold_0_array)
    features_test_ztautau = features_table(ztautau.fold_0_array)
    features_test_ttbar = features_table(ttbar.fold_0_array)
    
    print("Sample Event Input Features Before Scaling")
    print(features_test_HH_01[0])
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
    
    print("Scaler Info")
    print(scaler.mean_)
    print(scaler.scale_)
    
    log.info ('scaler ran')

    features_test_HH_01 = train_features_new[:len_HH_01]
    features_test_HH_10 = train_features_new[len_HH_01:len_HH_01+len_HH_10]
    features_test_ztautau = train_features_new[len_HH_01+len_HH_10:len_HH_01+len_HH_10+len_ztautau]
    features_test_ttbar = train_features_new[len_HH_01+len_HH_10+len_ztautau:]
    
    print("Sample Event Input Features After Scaling")
    print(features_test_HH_01[0])

    predictions_HH_01_0 = regressor_0.predict(features_test_HH_01)
    predictions_HH_10_0 = regressor_0.predict(features_test_HH_10)
    predictions_ztautau_0 = regressor_0.predict(features_test_ztautau)
    predictions_ttbar_0 = regressor_0.predict(features_test_ttbar)
    
    predictions_HH_01_1 = regressor_1.predict(features_test_HH_01)
    predictions_HH_10_1 = regressor_1.predict(features_test_HH_10)
    predictions_ztautau_1 = regressor_1.predict(features_test_ztautau)
    predictions_ttbar_1 = regressor_1.predict(features_test_ttbar)
    
    predictions_HH_01_2 = regressor_2.predict(features_test_HH_01)
    predictions_HH_10_2 = regressor_2.predict(features_test_HH_10)
    predictions_ztautau_2 = regressor_2.predict(features_test_ztautau)
    predictions_ttbar_2 = regressor_2.predict(features_test_ttbar)
    
    log.info ('regressor ran')
    log.info ('beginning fold combination')

    predictions_HH_01 = np.zeros(predictions_HH_01_0.shape)
    for i in range(predictions_HH_01.shape[0]):
        modulus = dihiggs_01.fold_0_array['EventInfo___NominalAuxDyn']['eventNumber'][i] % 3
        if (modulus == 0):
            predictions_HH_01[i,:] = predictions_HH_01_0[i,:]
        elif (modulus == 1):
            predictions_HH_01[i,:] = predictions_HH_01_1[i,:]
        else:
            predictions_HH_01[i,:] = predictions_HH_01_2[i,:]

    predictions_HH_10 = np.zeros(predictions_HH_10_0.shape)
    for i in range(predictions_HH_10.shape[0]):
        modulus = dihiggs_10.fold_0_array['EventInfo___NominalAuxDyn']['eventNumber'][i] % 3
        if (modulus == 0):
            predictions_HH_10[i,:] = predictions_HH_10_0[i,:]
        elif (modulus == 1):
            predictions_HH_10[i,:] = predictions_HH_10_1[i,:]
        else:
            predictions_HH_10[i,:] = predictions_HH_10_2[i,:]

    predictions_ztautau = np.zeros(predictions_ztautau_0.shape)
    for i in range(predictions_ztautau.shape[0]):
        modulus = ztautau.fold_0_array['EventInfo___NominalAuxDyn']['eventNumber'][i] % 3
        if (modulus == 0):
            predictions_ztautau[i,:] = predictions_ztautau_0[i,:]
        elif (modulus == 1):
            predictions_ztautau[i,:] = predictions_ztautau_1[i,:]
        else:
            predictions_ztautau[i,:] = predictions_ztautau_2[i,:]
            
    predictions_ttbar = np.zeros(predictions_ttbar_0.shape)
    for i in range(predictions_ttbar.shape[0]):
        modulus = ttbar.fold_0_array['EventInfo___NominalAuxDyn']['eventNumber'][i] % 3
        if (modulus == 0):
            predictions_ttbar[i,:] = predictions_ttbar_0[i,:]
        elif (modulus == 1):
            predictions_ttbar[i,:] = predictions_ttbar_1[i,:]
        else:
            predictions_ttbar[i,:] = predictions_ttbar_2[i,:]
    
    log.info ('done fold combination')
    log.info ('beginning reshaping')

    sigmas_HH_01 = np.exp(-0.5 * np.reshape(
        predictions_HH_01[:,1], (predictions_HH_01[:,1].shape[0], )))
    sigmas_HH_10 = np.exp(-0.5 * np.reshape(
        predictions_HH_10[:,1], (predictions_HH_10[:,1].shape[0], )))
    sigmas_ztautau = np.exp(-0.5 * np.reshape(
        predictions_ztautau[:,1], (predictions_ztautau[:,1].shape[0], )))
    sigmas_ttbar = np.exp(-0.5 * np.reshape(
        predictions_ttbar[:,1], (predictions_ttbar[:,1].shape[0], )))
        
    predictions_HH_01 = np.reshape(
        predictions_HH_01[:,0], (predictions_HH_01[:,0].shape[0], ))
    predictions_HH_10 = np.reshape(
        predictions_HH_10[:,0], (predictions_HH_10[:,0].shape[0], ))
    predictions_ztautau = np.reshape(
        predictions_ztautau[:,0], (predictions_ztautau[:,0].shape[0], ))
    predictions_ttbar = np.reshape(
        predictions_ttbar[:,0], (predictions_ttbar[:,0].shape[0], ))

    print("Sample Event Output")
    print(predictions_HH_01[0])
    print(sigmas_HH_01[0])

    log.info ('done reshaping')
    log.info ('beginning mvis computation')

    mvis_HH_01 = visable_mass(dihiggs_01.fold_0_array, 'dihiggs_01')
    mvis_HH_10 = visable_mass(dihiggs_10.fold_0_array, 'dihiggs_10')
    mvis_ztautau = visable_mass(ztautau.fold_0_array, 'ztautau')
    mvis_ttbar = visable_mass(ttbar.fold_0_array, 'ttbar')
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

    print (dihiggs_01.fold_0_array.fields)
    if 'mmc_bbtautau' in dihiggs_01.fold_0_array.fields:
        mmc_HH_01 = dihiggs_01.fold_0_array['mmc_tautau']
        mhh_mmc_HH_01 = dihiggs_01.fold_0_array['mmc_bbtautau']
    else:
        mmc_HH_01, mhh_mmc_HH_01 = mmc(dihiggs_01.fold_0_array)

    if 'mmc_bbtautau' in dihiggs_10.fold_0_array.fields: 
        mmc_HH_10 = dihiggs_10.fold_0_array['mmc_tautau']
        mhh_mmc_HH_10 = dihiggs_10.fold_0_array['mmc_bbtautau']
    else:
        mmc_HH_10, mhh_mmc_HH_10 = mmc(dihiggs_10.fold_0_array)

    if 'mmc_bbtautau' in ztautau.fold_0_array.fields:
        mmc_ztautau = ztautau.fold_0_array['mmc_tautau']
        mhh_mmc_ztautau = ztautau.fold_0_array['mmc_bbtautau']
    else:
        mmc_ztautau, mhh_mmc_ztautau = mmc(ztautau.fold_0_array)

    if 'mmc_bbtautau' in ttbar.fold_0_array.fields:
        mmc_ttbar = ttbar.fold_0_array['mmc_tautau']
        mhh_mmc_ttbar = ttbar.fold_0_array['mmc_bbtautau']
    else:
        mmc_ttbar, mhh_mmc_ttbar = mmc(ttbar.fold_0_array)
        
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
    for field in dihiggs_01.fold_0_array.fields:
        print(str(field))
        print(type(dihiggs_01.fold_0_array[field][0]))
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
    all_fold_0_arrays = np.concatenate([
        dihiggs_01.fold_0_array,
        dihiggs_10.fold_0_array,
        ztautau.fold_0_array,
        ttbar.fold_0_array
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
    
    #resol_HH_01 = resolution_plot(predictions_HH_01, sigmas_HH_01, dihiggs_01.fold_0_array, 'dihiggs_01')
    #resol_HH_10 = resolution_plot(predictions_HH_10, sigmas_HH_10, dihiggs_10.fold_0_array, 'dihiggs_10')
    #resol_ztautau = resolution_plot(predictions_ztautau, sigmas_ztautau, ztautau.fold_0_array, 'ztautau')
    #resol_ttbar = resolution_plot(predictions_ttbar, sigmas_ttbar, ttbar.fold_0_array, 'ttbar')
    #resol_all = resolution_plot(all_predictions, all_sigmas, all_fold_0_arrays, 'all')
    #res_plots(all_predictions, all_sigmas, all_fold_0_arrays, 'all')
    signal_predictions = np.concatenate([
        predictions_HH_01,
        predictions_HH_10
    ])
    signal_sigmas = np.concatenate([
        sigmas_HH_01,
        sigmas_HH_10
    ])
    signal_fold_0_arrays = np.concatenate([
        dihiggs_01.fold_0_array,
        dihiggs_10.fold_0_array
    ])
    #resol_signal = resolution_plot(signal_predictions, signal_sigmas, signal_fold_0_arrays, 'signal')
    #res_plots(signal_predictions, signal_sigmas, signal_fold_0_arrays, 'signal')
    background_predictions = np.concatenate([
        predictions_ztautau,
        predictions_ttbar
    ])
    background_sigmas = np.concatenate([
        sigmas_ztautau,
        sigmas_ttbar
    ])
    background_fold_0_arrays = np.concatenate([
        ztautau.fold_0_array,
        ttbar.fold_0_array
    ])
    #resol_background = resolution_plot(background_predictions, background_sigmas, background_fold_0_arrays, 'background')
    #res_plots(background_predictions, background_sigmas, background_fold_0_arrays, 'background')

    """
    resol_HH_01 = resol_all[:len_HH_01]
    resol_HH_10 = resol_all[len_HH_01:len_HH_01+len_HH_10]
    resol_ztautau = resol_all[len_HH_01+len_HH_10:len_HH_01+len_HH_10+len_ztautau]
    resol_ttbar = resol_all[len_HH_01+len_HH_10+len_ztautau:]
    resol_HH_01 = resol_signal[:len_HH_01]
    resol_HH_10 = resol_signal[len_HH_01:]
    resol_ztautau = resol_background[:len_ztautau]
    resol_ttbar = resol_background[len_ztautau:]
    resol_all = np.concatenate([
        resol_signal,
        resol_background
    ])
    """
    
    log.info ('Beginning Sigma Plotting')
    
    """
    # Use this to define the cut on a constant rather than resolution as a function of mHH
    split_const = 0.2
    resol_HH_01 = split_const
    resol_HH_10 = split_const
    resol_ztautau = split_const
    resol_ttbar = split_const
    resol_all = split_const
    resol_signal = split_const
    resol_background = split_const
    """
    
    index_dict = {'HH01':{}, 'HH10':{}, 'ztautau':{}, 'ttbar':{}, 'all':{}}
    
    index_dict['HH01'][0] = np.where(sigmas_HH_01 / predictions_HH_01 < 0.075)
    index_dict['HH10'][0] = np.where(sigmas_HH_10 / predictions_HH_10 < 0.075)
    index_dict['ztautau'][0] = np.where(sigmas_ztautau / predictions_ztautau < 0.075)
    index_dict['ttbar'][0] = np.where(sigmas_ttbar / predictions_ttbar < 0.075)
    index_dict['all'][0] = np.where(all_sigmas / all_predictions < 0.075)
    
    index_dict['HH01'][1] = np.where((sigmas_HH_01 / predictions_HH_01 > 0.075) & (sigmas_HH_01 / predictions_HH_01 < 0.085))
    index_dict['HH10'][1] = np.where((sigmas_HH_10 / predictions_HH_10 > 0.075) & (sigmas_HH_10 / predictions_HH_10 < 0.085))
    index_dict['ztautau'][1] = np.where((sigmas_ztautau / predictions_ztautau > 0.075) & (sigmas_ztautau / predictions_ztautau < 0.085))
    index_dict['ttbar'][1] = np.where((sigmas_ttbar / predictions_ttbar > 0.075) & (sigmas_ttbar / predictions_ttbar < 0.085))
    index_dict['all'][1] = np.where((all_sigmas / all_predictions > 0.075) & (all_sigmas / all_predictions < 0.085))
    
    index_dict['HH01'][2] = np.where((sigmas_HH_01 / predictions_HH_01 > 0.085) & (sigmas_HH_01 / predictions_HH_01 < 0.12))
    index_dict['HH10'][2] = np.where((sigmas_HH_10 / predictions_HH_10 > 0.085) & (sigmas_HH_10 / predictions_HH_10 < 0.12))
    index_dict['ztautau'][2] = np.where((sigmas_ztautau / predictions_ztautau > 0.085) & (sigmas_ztautau / predictions_ztautau < 0.12))
    index_dict['ttbar'][2] = np.where((sigmas_ttbar / predictions_ttbar > 0.085) & (sigmas_ttbar / predictions_ttbar < 0.12))
    index_dict['all'][2] = np.where((all_sigmas / all_predictions > 0.085) & (all_sigmas / all_predictions < 0.12))
    
    index_dict['HH01'][3] = np.where((sigmas_HH_01 / predictions_HH_01 > 0.12) & (sigmas_HH_01 / predictions_HH_01 < 0.18))
    index_dict['HH10'][3] = np.where((sigmas_HH_10 / predictions_HH_10 > 0.12) & (sigmas_HH_10 / predictions_HH_10 < 0.18))
    index_dict['ztautau'][3] = np.where((sigmas_ztautau / predictions_ztautau > 0.12) & (sigmas_ztautau / predictions_ztautau < 0.18))
    index_dict['ttbar'][3] = np.where((sigmas_ttbar / predictions_ttbar > 0.12) & (sigmas_ttbar / predictions_ttbar < 0.18))
    index_dict['all'][3] = np.where((all_sigmas / all_predictions > 0.12) & (all_sigmas / all_predictions < 0.18))
    
    index_dict['HH01'][4] = np.where(sigmas_HH_01 / predictions_HH_01 > 0.18)
    index_dict['HH10'][4] = np.where(sigmas_HH_10 / predictions_HH_10 > 0.18)
    index_dict['ztautau'][4] = np.where(sigmas_ztautau / predictions_ztautau > 0.18)
    index_dict['ttbar'][4] = np.where(sigmas_ttbar / predictions_ttbar > 0.18)
    index_dict['all'][4] = np.where(all_sigmas / all_predictions > 0.18)
    
    """
    indices_1_1 = np.where(sigmas_HH_01 / predictions_HH_01 < resol_HH_01)
    indices_1_10 = np.where(sigmas_HH_10 / predictions_HH_10 < resol_HH_10)
    indices_1_z = np.where(sigmas_ztautau / predictions_ztautau < resol_ztautau)
    indices_1_t = np.where(sigmas_ttbar / predictions_ttbar < resol_ttbar)
    indices_2_1 = np.where(sigmas_HH_01 / predictions_HH_01 > resol_HH_01)
    indices_2_10 = np.where(sigmas_HH_10 / predictions_HH_10 > resol_HH_10)
    indices_2_z = np.where(sigmas_ztautau / predictions_ztautau > resol_ztautau)
    indices_2_t = np.where(sigmas_ttbar / predictions_ttbar > resol_ttbar)
    
    indices_1_all = np.where(all_sigmas / all_predictions < resol_all)
    indices_2_all = np.where(all_sigmas / all_predictions > resol_all)
    
    sigma_plots(predictions_HH_01, sigmas_HH_01, dihiggs_01.fold_0_array, 'dihiggs_01', mvis_HH_01, indices_1=indices_1_1, indices_2=indices_2_1)
    sigma_plots(predictions_HH_10, sigmas_HH_10, dihiggs_10.fold_0_array, 'dihiggs_10', mvis_HH_10, indices_1=indices_1_10, indices_2=indices_2_10)
    sigma_plots(predictions_ztautau, sigmas_ztautau, ztautau.fold_0_array, 'ztautau', mvis_ztautau, indices_1=indices_1_z, indices_2=indices_2_z)
    sigma_plots(predictions_ttbar, sigmas_ttbar, ttbar.fold_0_array, 'ttbar', mvis_ttbar, indices_1=indices_1_t, indices_2=indices_2_t)
    sigma_plots(all_predictions, all_sigmas, all_fold_0_arrays, 'all', all_mvis, indices_1=indices_1_all, indices_2=indices_2_all)
    
    indices_signal_low = np.where((signal_sigmas / signal_predictions) < resol_signal)
    indices_signal_high = np.where((signal_sigmas / signal_predictions) > resol_signal)
    indices_background_low = np.where((background_sigmas / background_predictions) < resol_background)
    indices_background_high = np.where((background_sigmas / background_predictions) > resol_background)
    
    resolution_plot(signal_predictions[indices_signal_low], signal_sigmas[indices_signal_low], signal_fold_0_arrays[indices_signal_low], 'signal_sigma_less_than_resol')
    resolution_plot(signal_predictions[indices_signal_high], signal_sigmas[indices_signal_high], signal_fold_0_arrays[indices_signal_high], 'signal_sigma_greater_than_resol')
    resolution_plot(background_predictions[indices_background_low], background_sigmas[indices_background_low], background_fold_0_arrays[indices_background_low], 'background_sigma_less_than_resol')
    resolution_plot(background_predictions[indices_background_high], background_sigmas[indices_background_high], background_fold_0_arrays[indices_background_high], 'background_sigma_greater_than_resol')
    res_plots(signal_predictions[indices_signal_low], signal_sigmas[indices_signal_low], signal_fold_0_arrays[indices_signal_low], 'signal_sigma_less_than_resol')
    res_plots(signal_predictions[indices_signal_high], signal_sigmas[indices_signal_high], signal_fold_0_arrays[indices_signal_high], 'signal_sigma_greater_than_resol')
    res_plots(background_predictions[indices_background_low], background_sigmas[indices_background_low], background_fold_0_arrays[indices_background_low], 'background_sigma_less_than_resol')
    res_plots(background_predictions[indices_background_high], background_sigmas[indices_background_high], background_fold_0_arrays[indices_background_high], 'background_sigma_greater_than_resol')
    
    # Use this to split by resolution rather than sigma
    sigmas_HH_01 /= predictions_HH_01 * resol_HH_01
    sigmas_HH_10 /= predictions_HH_10 * resol_HH_10
    sigmas_ztautau /= predictions_ztautau * resol_ztautau
    sigmas_ttbar /= predictions_ttbar * resol_ttbar
    all_sigmas = np.concatenate([
        sigmas_HH_01,
        sigmas_HH_10,
        sigmas_ztautau,
        sigmas_ttbar
    ])
    """
    
    simple_sigma_plot(sigmas_HH_01, dihiggs_01.fold_0_array, 'dihiggs_01')
    simple_sigma_plot(sigmas_HH_10, dihiggs_10.fold_0_array, 'dihiggs_10')
    simple_sigma_plot(sigmas_ztautau, ztautau.fold_0_array, 'ztautau')
    simple_sigma_plot(sigmas_ttbar, ttbar.fold_0_array, 'ttbar')
    simple_sigma_plot(all_sigmas, all_fold_0_arrays, 'all',)
    
    print('How many events in each split category:')
    for i in range(5):
        print(i)
        print(len(index_dict['HH01'][i]))
        print(len(index_dict['HH10'][i]))
        print(len(index_dict['ztautau'][i]))
        print(len(index_dict['ttbar'][i]))
    
    log.info ('Beginning Sigma Plotting')
    
    sigma_plots(predictions_HH_01, sigmas_HH_01, dihiggs_01.fold_0_array, 'dihiggs_01', mvis_HH_01)
    sigma_plots(predictions_HH_10, sigmas_HH_10, dihiggs_10.fold_0_array, 'dihiggs_10', mvis_HH_10)
    sigma_plots(predictions_ztautau, sigmas_ztautau, ztautau.fold_0_array, 'ztautau', mvis_ztautau)
    sigma_plots(predictions_ttbar, sigmas_ttbar, ttbar.fold_0_array, 'ttbar', mvis_ttbar)
    sigma_plots(all_predictions, all_sigmas, all_fold_0_arrays, 'all', all_mvis)
    
    """
    resid_comparison_plots(predictions_HH_01, sigmas_HH_01, mhh_original_HH_01, mhh_mmc_HH_01, dihiggs_01.fold_0_array, 'dihiggs_01', mvis_HH_01)
    resid_comparison_plots(predictions_HH_10, sigmas_HH_10, mhh_original_HH_10, mhh_mmc_HH_10, dihiggs_10.fold_0_array, 'dihiggs_10', mvis_HH_10)
    resid_comparison_plots(predictions_ztautau, sigmas_ztautau, mhh_original_ztautau, mhh_mmc_ztautau, ztautau.fold_0_array, 'ztautau', mvis_ztautau)
    resid_comparison_plots(predictions_ttbar, sigmas_ttbar, mhh_original_ttbar, mhh_mmc_ttbar, ttbar.fold_0_array, 'ttbar', mvis_ttbar)
    resid_comparison_plots(all_predictions, all_sigmas, all_original, all_mmc, all_fold_0_arrays, 'all', all_mvis)

    
    log.info ('Finished Sigma Plotting, Beginning k_lambda Comparison Plotting')
    
    k_lambda_comparison_plot(dihiggs_01.fold_0_array['universal_true_mhh'], dihiggs_10.fold_0_array['universal_true_mhh'], dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'truth')
    k_lambda_comparison_plot(predictions_HH_01, predictions_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'mdn')
    k_lambda_comparison_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'mmc')
    k_lambda_comparison_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'mmc_fail', slice_indices_1 = np.where(mmc_HH_01 == 0), slice_indices_10 = np.where(mmc_HH_10 == 0))
    k_lambda_comparison_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'mmc_pass', slice_indices_1 = np.where(mmc_HH_01 > 0), slice_indices_10 = np.where(mmc_HH_10 > 0))
    k_lambda_comparison_plot(mhh_original_HH_01, mhh_original_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'dnn')
    
    k_lambda_comparison_plot(predictions_HH_01[indices_1_1], predictions_HH_10[indices_1_10], dihiggs_01.fold_0_array[indices_1_1], dihiggs_10.fold_0_array[indices_1_10], 'mdn_low_sigma')
    k_lambda_comparison_plot(predictions_HH_01[indices_2_1], predictions_HH_10[indices_2_10], dihiggs_01.fold_0_array[indices_2_1], dihiggs_10.fold_0_array[indices_2_10], 'mdn_high_sigma')
    """
   
    log.info ('Finished k_lambda Comparison Plotting, Beginning RNN-MMC Comparison Plotting')
    
    """
    eff_HH_01_rnn_mmc, eff_true_HH_01, n_rnn_HH_01, n_mmc_HH_01, n_true_HH_01 = rnn_mmc_comparison(predictions_HH_01, test_target_HH_01, dihiggs_01, dihiggs_01.fold_0_array, 'dihiggs_01', args.library, predictions_old = mhh_original_HH_01, predictions_mmc = mhh_mmc_HH_01)
    eff_HH_10_rnn_mmc, eff_true_HH_10, n_rnn_HH_10, n_mmc_HH_10, n_true_HH_10 = rnn_mmc_comparison(predictions_HH_10, test_target_HH_10, dihiggs_10, dihiggs_10.fold_0_array, 'dihiggs_10', args.library, predictions_old = mhh_original_HH_10, predictions_mmc = mhh_mmc_HH_10)
    eff_ztt_rnn_mmc, eff_true_ztt, n_rnn_ztt, n_mmc_ztt, n_true_ztt = rnn_mmc_comparison(predictions_ztautau, test_target_ztautau, ztautau, ztautau.fold_0_array, 'ztautau', args.library, predictions_old = mhh_original_ztautau, predictions_mmc = mhh_mmc_ztautau)
    eff_ttbar_rnn_mmc, eff_true_ttbar, n_rnn_ttbar, n_mmc_ttbar, n_true_ttbar = rnn_mmc_comparison(predictions_ttbar, test_target_ttbar, ttbar, ttbar.fold_0_array, 'ttbar', args.library, predictions_old = mhh_original_ttbar, predictions_mmc = mhh_mmc_ttbar)
    """
    """
    eff_HH_01_rnn_mmc, eff_true_HH_01, n_rnn_HH_01, n_mmc_HH_01, n_true_HH_01 = rnn_mmc_comparison(predictions_HH_01, test_target_HH_01, dihiggs_01, dihiggs_01.fold_0_array, 'dihiggs_01', args.library, predictions_mmc = mhh_mmc_HH_01)
    eff_HH_10_rnn_mmc, eff_true_HH_10, n_rnn_HH_10, n_mmc_HH_10, n_true_HH_10 = rnn_mmc_comparison(predictions_HH_10, test_target_HH_10, dihiggs_10, dihiggs_10.fold_0_array, 'dihiggs_10', args.library, predictions_mmc = mhh_mmc_HH_10)
    eff_ztt_rnn_mmc, eff_true_ztt, n_rnn_ztt, n_mmc_ztt, n_true_ztt = rnn_mmc_comparison(predictions_ztautau, test_target_ztautau, ztautau, ztautau.fold_0_array, 'ztautau', args.library, predictions_mmc = mhh_mmc_ztautau)
    eff_ttbar_rnn_mmc, eff_true_ttbar, n_rnn_ttbar, n_mmc_ttbar, n_true_ttbar = rnn_mmc_comparison(predictions_ttbar, test_target_ttbar, ttbar, ttbar.fold_0_array, 'ttbar', args.library, predictions_mmc = mhh_mmc_ttbar)
    """

    log.info ('Finished RNN-MMC Comparison Plotting')

    # Chi-Square calculations

    # Relevant ROC curves
    """
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
    """
    
    """
    log.info ('Beginning Sigma-Split RNN-MMC Comparison Plotting')
    
    rnn_mmc_comparison(all_predictions, all_fold_0_arrays['universal_true_mhh'], dihiggs_01, all_fold_0_arrays, 'all', args.library, predictions_mmc = all_mmc)
    rnn_mmc_comparison(all_predictions, all_fold_0_arrays['universal_true_mhh'], dihiggs_01, all_fold_0_arrays, 'all', args.library, predictions_mmc = all_mmc, sigma_label = '_low_sigma', sigma_slice = indices_1_all)
    rnn_mmc_comparison(all_predictions, all_fold_0_arrays['universal_true_mhh'], dihiggs_01, all_fold_0_arrays, 'all', args.library, predictions_mmc = all_mmc, sigma_label = '_high_sigma', sigma_slice = indices_2_all)
    
    # lowest sigma
    eff_HH_01_rnn_mmc, eff_true_HH_01, n_rnn_HH_01, n_mmc_HH_01, n_true_HH_01 = rnn_mmc_comparison(predictions_HH_01, test_target_HH_01, dihiggs_01, dihiggs_01.fold_0_array, 'dihiggs_01', args.library, predictions_old = mhh_original_HH_01, predictions_mmc = mhh_mmc_HH_01, sigma_label = '_low_sigma', sigma_slice = indices_1_1)
    eff_HH_10_rnn_mmc, eff_true_HH_10, n_rnn_HH_10, n_mmc_HH_10, n_true_HH_10 = rnn_mmc_comparison(predictions_HH_10, test_target_HH_10, dihiggs_10, dihiggs_10.fold_0_array, 'dihiggs_10', args.library, predictions_old = mhh_original_HH_10, predictions_mmc = mhh_mmc_HH_10, sigma_label = '_low_sigma', sigma_slice = indices_1_10)
    eff_ztt_rnn_mmc, eff_true_ztt, n_rnn_ztt, n_mmc_ztt, n_true_ztt = rnn_mmc_comparison(predictions_ztautau, test_target_ztautau, ztautau, ztautau.fold_0_array, 'ztautau', args.library, predictions_old = mhh_original_ztautau, predictions_mmc = mhh_mmc_ztautau, sigma_label = '_low_sigma', sigma_slice = indices_1_z)
    eff_ttbar_rnn_mmc, eff_true_ttbar, n_rnn_ttbar, n_mmc_ttbar, n_true_ttbar = rnn_mmc_comparison(predictions_ttbar, test_target_ttbar, ttbar, ttbar.fold_0_array, 'ttbar', args.library, predictions_old = mhh_original_ttbar, predictions_mmc = mhh_mmc_ttbar, sigma_label = '_low_sigma', sigma_slice = indices_1_t)
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
    eff_HH_01_rnn_mmc, eff_true_HH_01, n_rnn_HH_01, n_mmc_HH_01, n_true_HH_01 = rnn_mmc_comparison(predictions_HH_01, test_target_HH_01, dihiggs_01, dihiggs_01.fold_0_array, 'dihiggs_01', args.library, predictions_old = mhh_original_HH_01, predictions_mmc = mhh_mmc_HH_01, sigma_label = '_high_sigma', sigma_slice = indices_2_1)
    eff_HH_10_rnn_mmc, eff_true_HH_10, n_rnn_HH_10, n_mmc_HH_10, n_true_HH_10 = rnn_mmc_comparison(predictions_HH_10, test_target_HH_10, dihiggs_10, dihiggs_10.fold_0_array, 'dihiggs_10', args.library, predictions_old = mhh_original_HH_10, predictions_mmc = mhh_mmc_HH_10, sigma_label = '_high_sigma', sigma_slice = indices_2_10)
    eff_ztt_rnn_mmc, eff_true_ztt, n_rnn_ztt, n_mmc_ztt, n_true_ztt = rnn_mmc_comparison(predictions_ztautau, test_target_ztautau, ztautau, ztautau.fold_0_array, 'ztautau', args.library, predictions_old = mhh_original_ztautau, predictions_mmc = mhh_mmc_ztautau, sigma_label = '_high_sigma', sigma_slice = indices_2_z)
    eff_ttbar_rnn_mmc, eff_true_ttbar, n_rnn_ttbar, n_mmc_ttbar, n_true_ttbar = rnn_mmc_comparison(predictions_ttbar, test_target_ttbar, ttbar, ttbar.fold_0_array, 'ttbar', args.library, predictions_old = mhh_original_ttbar, predictions_mmc = mhh_mmc_ttbar, sigma_label = '_high_sigma', sigma_slice = indices_2_t)
    eff_pred_HH_01_HH_10 = eff_HH_01_rnn_mmc + eff_HH_10_rnn_mmc
    eff_pred_HH_01_ztt = eff_HH_01_rnn_mmc + eff_ztt_rnn_mmc
    eff_pred_HH_01_ttbar = eff_HH_01_rnn_mmc + eff_ttbar_rnn_mmc
    eff_true_HH_01_HH_10 = [eff_true_HH_01] + [eff_true_HH_10]
    eff_true_HH_01_ztt = [eff_true_HH_01] + [eff_true_ztt]
    eff_true_HH_01_ttbar = [eff_true_HH_01] + [eff_true_ttbar]
    roc_plot_rnn_mmc(eff_pred_HH_01_HH_10, eff_true_HH_01_HH_10, r'$\kappa_{\lambda}$ = 1', r'$\kappa_{\lambda}$ = 10', sigma_label = '_high_sigma')
    roc_plot_rnn_mmc(eff_pred_HH_01_ztt, eff_true_HH_01_ztt, r'$\kappa_{\lambda}$ = 1', r'$Z\to\tau\tau$ + jets', sigma_label = '_high_sigma')
    roc_plot_rnn_mmc(eff_pred_HH_01_ttbar, eff_true_HH_01_ttbar, r'$\kappa_{\lambda}$ = 1', 'Top Quark', sigma_label = '_high_sigma')
    """

    """
    # Pile-up stability of the signal
    avg_mhh_HH_01 = avg_mhh_calculation(dihiggs_01.fold_0_array, test_target_HH_01, predictions_HH_01, mhh_mmc_HH_01)
    avg_mhh_HH_10 = avg_mhh_calculation(dihiggs_10.fold_0_array, test_target_HH_10, predictions_HH_10, mhh_mmc_HH_10)
    avg_mhh_plot(avg_mhh_HH_01, 'pileup_stability_avg_mhh_HH_01', dihiggs_01)
    avg_mhh_plot(avg_mhh_HH_10, 'pileup_stability_avg_mhh_HH_10', dihiggs_10)
    """
    
    # Attempt to reweight klambda=1 to klambda=10
    log.info('Loading Reweight Root Files')
    reweight_file_1 = uproot.open("data/weight-mHH-from-cHHHp01d0-to-cHHHpx_20GeV_Jul28.root")
    reweight = reweight_file_1["reweight_mHH_1p0_to_10p0"].to_numpy()
    norm = reweight_file_1["norm10p0"].value
    """
    reweight_file_10 = uproot.open("data/weight-mHH-from-cHHHp10d0-to-cHHHpx_20GeV_Jul28.root")
    reweight_10 = reweight_file_10["reweight_mHH_1p0_to_1p0"].to_numpy()
    """
    log.info('Beginning Reweight Plots')
    
    reweights_by_bin = reweight[0] * norm
    num_bins = len(reweights_by_bin)
    new_weights = []
    for i in range(len(dihiggs_01.fold_0_array['universal_true_mhh'])):
        reweight_bin = int((dihiggs_01.fold_0_array['universal_true_mhh'][i] - 200) / 20)
        if ((reweight_bin > -1) and (reweight_bin < num_bins)):
            new_weights.append(dihiggs_01.fold_0_array['EventInfo___NominalAuxDyn']['evtweight'][i] * dihiggs_01.fold_0_array['fold_weight'][i] * reweights_by_bin[reweight_bin])
        else:
            new_weights.append(0)
    new_weights = np.array(new_weights)
    reweight_plot(dihiggs_01.fold_0_array['universal_true_mhh'], dihiggs_10.fold_0_array['universal_true_mhh'], dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, new_weights, 'truth')
    reweight_plot(predictions_HH_01, predictions_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, new_weights, 'mdn')
    reweight_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, new_weights, 'mmc')
    
    """
    eta_plot(dihiggs_01.fold_0_array['higgs'], np.array(dihiggs_01.fold_0_array['EventInfo___NominalAuxDyn']['evtweight'] * dihiggs_01.fold_0_array['fold_weight']), 'HH_01_original')
    eta_plot(dihiggs_10.fold_0_array['higgs'], np.array(dihiggs_10.fold_0_array['EventInfo___NominalAuxDyn']['evtweight'] * dihiggs_10.fold_0_array['fold_weight']), 'HH_10_original')
    eta_plot(dihiggs_01.fold_0_array['higgs'], new_weights, 'HH_10_reweighted')
    """
    
    # Scan over a range of klambda values and find their significances
    klambda_scan_list = ['n8p0', 'n7p0', 'n6p0', 'n5p0', 'n4p0', 'n3p0', 'n2p0', 'n1p0', '0p0', '1p0', '2p0', '3p0', '4p0', '5p0', '6p0', '7p0', '8p0', '9p0', '10p0', '11p0', '12p0', '13p0', '14p0', '15p0']
    reweight_file = uproot.open("data/weight-mHH-from-cHHHp01d0-to-cHHHpx_20GeV_Jul28.root")
    
    truth_significances = []
    mdn_significances = []
    mmc_significances = []
    split_significances = []
    original_significances = []
    split_truth = []
    original_weights = dihiggs_01.fold_0_array['EventInfo___NominalAuxDyn']['evtweight'] * dihiggs_01.fold_0_array['fold_weight']
    
    for klambda in klambda_scan_list:
    
        print('klambda reweight scan: ' + klambda)
        
        if (klambda == '1p0'):
            truth_significances.append(0)
            mdn_significances.append(0)
            mmc_significances.append(0)
            split_significances.append(0)
            split_truth.append(0)
            continue
            
        reweight = reweight_file['reweight_mHH_1p0_to_' + klambda].to_numpy()
        norm = reweight_file['norm' + klambda].value
        
        reweights_by_bin = reweight[0] * norm
        num_bins = len(reweights_by_bin)
        
        new_weights = []
        for i in range(len(dihiggs_01.fold_0_array['universal_true_mhh'])):
            reweight_bin = int((dihiggs_01.fold_0_array['universal_true_mhh'][i] - 200) / 20)
            if ((reweight_bin > -1) and (reweight_bin < num_bins)):
                new_weights.append(original_weights[i] * reweights_by_bin[reweight_bin])
            else:
                new_weights.append(0)
        new_weights = np.array(new_weights)
                
        z, cs_norm = reweight_and_compare(dihiggs_01.fold_0_array['universal_true_mhh'], original_weights, new_weights, 'truth', klambda)
        truth_significances.append(z)
        
        z, cs_norm = reweight_and_compare(predictions_HH_01, original_weights, new_weights, 'mdn', klambda, cs_norm = cs_norm)
        mdn_significances.append(z)
        
        z, cs_norm = reweight_and_compare(mhh_mmc_HH_01, original_weights, new_weights, 'mmc', klambda, cs_norm = cs_norm)
        mmc_significances.append(z)
        
        z0, cs_norm = reweight_and_compare(predictions_HH_01[index_dict['HH01'][0]], original_weights[index_dict['HH01'][0]], new_weights[index_dict['HH01'][0]], 'mdn_split_0', klambda, cs_norm = cs_norm)
        z1, cs_norm = reweight_and_compare(predictions_HH_01[index_dict['HH01'][1]], original_weights[index_dict['HH01'][1]], new_weights[index_dict['HH01'][1]], 'mdn_split_1', klambda, cs_norm = cs_norm)
        z2, cs_norm = reweight_and_compare(predictions_HH_01[index_dict['HH01'][2]], original_weights[index_dict['HH01'][2]], new_weights[index_dict['HH01'][2]], 'mdn_split_2', klambda, cs_norm = cs_norm)
        z3, cs_norm = reweight_and_compare(predictions_HH_01[index_dict['HH01'][3]], original_weights[index_dict['HH01'][3]], new_weights[index_dict['HH01'][3]], 'mdn_split_3', klambda, cs_norm = cs_norm)
        z4, cs_norm = reweight_and_compare(predictions_HH_01[index_dict['HH01'][4]], original_weights[index_dict['HH01'][4]], new_weights[index_dict['HH01'][4]], 'mdn_split_4', klambda, cs_norm = cs_norm)
        z = np.sqrt(z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4)
        print("MDN Splitting Improvement: " + str(z - mdn_significances[-1]))
        split_significances.append(z)
        
        z0, cs_norm = reweight_and_compare(dihiggs_01.fold_0_array[index_dict['HH01'][0]]['universal_true_mhh'], original_weights[index_dict['HH01'][0]], new_weights[index_dict['HH01'][0]], 'truth_split_0', klambda, cs_norm = cs_norm)
        z1, cs_norm = reweight_and_compare(dihiggs_01.fold_0_array[index_dict['HH01'][1]]['universal_true_mhh'], original_weights[index_dict['HH01'][1]], new_weights[index_dict['HH01'][1]], 'truth_split_1', klambda, cs_norm = cs_norm)
        z2, cs_norm = reweight_and_compare(dihiggs_01.fold_0_array[index_dict['HH01'][2]]['universal_true_mhh'], original_weights[index_dict['HH01'][2]], new_weights[index_dict['HH01'][2]], 'truth_split_2', klambda, cs_norm = cs_norm)
        z3, cs_norm = reweight_and_compare(dihiggs_01.fold_0_array[index_dict['HH01'][3]]['universal_true_mhh'], original_weights[index_dict['HH01'][3]], new_weights[index_dict['HH01'][3]], 'truth_split_3', klambda, cs_norm = cs_norm)
        z4, cs_norm = reweight_and_compare(dihiggs_01.fold_0_array[index_dict['HH01'][4]]['universal_true_mhh'], original_weights[index_dict['HH01'][4]], new_weights[index_dict['HH01'][4]], 'truth_split_4', klambda, cs_norm = cs_norm)
        z = np.sqrt(z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4)
        print("Truth Splitting Improvement: " + str(z - truth_significances[-1]))
        split_truth.append(z)
    
    z = k_lambda_comparison_plot(dihiggs_01.fold_0_array['universal_true_mhh'], dihiggs_10.fold_0_array['universal_true_mhh'], dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'truth')
    original_significances.append(z)
    z = k_lambda_comparison_plot(predictions_HH_01, predictions_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'mdn')
    reader_file = uproot.open("cache/analysis_ade Signal Only.root")
    k01_reader = reader_file["HHMassNet/hhttbb_2tag_0mHH_LL_OS_GGFSR_HHMassNet_mass;1"].to_numpy()
    k10_reader = reader_file["HHMassNet/hhttbbL10_2tag_0mHH_LL_OS_GGFSR_HHMassNet_mass;1"].to_numpy()
    k_lambda_comparison_reader_overlay(predictions_HH_01, predictions_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'mdn', k01_reader, k10_reader)
    original_significances.append(z)
    z = k_lambda_comparison_plot(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'mmc')
    k01_reader = reader_file["Preselection/hhttbb_2tag_0mHH_LL_OS_GGFSR_mHH;1"].to_numpy()
    k10_reader = reader_file["Preselection/hhttbbL10_2tag_0mHH_LL_OS_GGFSR_mHH;1"].to_numpy()
    k_lambda_comparison_reader_overlay(mhh_mmc_HH_01, mhh_mmc_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, 'mmc', k01_reader, k10_reader)
    original_significances.append(z)
    z0 = k_lambda_comparison_plot(predictions_HH_01[index_dict['HH01'][0]], predictions_HH_10[index_dict['HH10'][0]], dihiggs_01.fold_0_array[index_dict['HH01'][0]], dihiggs_10.fold_0_array[index_dict['HH10'][0]], 'mdn_split_0')
    z1 = k_lambda_comparison_plot(predictions_HH_01[index_dict['HH01'][1]], predictions_HH_10[index_dict['HH10'][1]], dihiggs_01.fold_0_array[index_dict['HH01'][1]], dihiggs_10.fold_0_array[index_dict['HH10'][1]], 'mdn_split_1')
    z2 = k_lambda_comparison_plot(predictions_HH_01[index_dict['HH01'][2]], predictions_HH_10[index_dict['HH10'][2]], dihiggs_01.fold_0_array[index_dict['HH01'][2]], dihiggs_10.fold_0_array[index_dict['HH10'][2]], 'mdn_split_2')
    z3 = k_lambda_comparison_plot(predictions_HH_01[index_dict['HH01'][3]], predictions_HH_10[index_dict['HH10'][3]], dihiggs_01.fold_0_array[index_dict['HH01'][3]], dihiggs_10.fold_0_array[index_dict['HH10'][3]], 'mdn_split_3')
    z4 = k_lambda_comparison_plot(predictions_HH_01[index_dict['HH01'][4]], predictions_HH_10[index_dict['HH10'][4]], dihiggs_01.fold_0_array[index_dict['HH01'][4]], dihiggs_10.fold_0_array[index_dict['HH10'][4]], 'mdn_split_4')
    z = np.sqrt(z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4)
    original_significances.append(z)
    z0 = k_lambda_comparison_plot(dihiggs_01.fold_0_array[index_dict['HH01'][0]]['universal_true_mhh'], dihiggs_10.fold_0_array[index_dict['HH10'][0]]['universal_true_mhh'], dihiggs_01.fold_0_array[index_dict['HH01'][0]], dihiggs_10.fold_0_array[index_dict['HH10'][0]], 'truth_split_0')
    z1 = k_lambda_comparison_plot(dihiggs_01.fold_0_array[index_dict['HH01'][1]]['universal_true_mhh'], dihiggs_10.fold_0_array[index_dict['HH10'][1]]['universal_true_mhh'], dihiggs_01.fold_0_array[index_dict['HH01'][1]], dihiggs_10.fold_0_array[index_dict['HH10'][1]], 'truth_split_1')
    z2 = k_lambda_comparison_plot(dihiggs_01.fold_0_array[index_dict['HH01'][2]]['universal_true_mhh'], dihiggs_10.fold_0_array[index_dict['HH10'][2]]['universal_true_mhh'], dihiggs_01.fold_0_array[index_dict['HH01'][2]], dihiggs_10.fold_0_array[index_dict['HH10'][2]], 'truth_split_2')
    z3 = k_lambda_comparison_plot(dihiggs_01.fold_0_array[index_dict['HH01'][3]]['universal_true_mhh'], dihiggs_10.fold_0_array[index_dict['HH10'][3]]['universal_true_mhh'], dihiggs_01.fold_0_array[index_dict['HH01'][3]], dihiggs_10.fold_0_array[index_dict['HH10'][3]], 'truth_split_3')
    z4 = k_lambda_comparison_plot(dihiggs_01.fold_0_array[index_dict['HH01'][4]]['universal_true_mhh'], dihiggs_10.fold_0_array[index_dict['HH10'][4]]['universal_true_mhh'], dihiggs_01.fold_0_array[index_dict['HH01'][4]], dihiggs_10.fold_0_array[index_dict['HH10'][4]], 'truth_split_4')
    z = np.sqrt(z0 * z0 + z1 * z1 + z2 * z2 + z3 * z3 + z4 * z4)
    original_significances.append(z)
    
    #separation_overlay_plot(predictions_HH_01, predictions_HH_10, dihiggs_01.fold_0_array, dihiggs_10.fold_0_array, indices_1_1, indices_1_10, indices_2_1, indices_2_10)
    klambda_scan_plot(range(-8, 16), truth_significances, mdn_significances, mmc_significances, split_significances, bonus_pts = original_significances, split_truth = split_truth)
    #klambda_scan_plot(range(-8, 16), truth_significances, mdn_significances, mmc_significances, split_significances, bonus_pts = original_significances)
    
    """
    # Scan over a range of klambda values and find their significances again, but starting from klambda=10
    reweight_file = uproot.open("data/weight-mHH-from-cHHHp10d0-to-cHHHpx_20GeV_Jul28.root")
    
    truth_significances = []
    mdn_significances = []
    mmc_significances = []
    split_significances = []
    original_significances = []
    split_truth = []
    original_weights = dihiggs_10.fold_0_array['EventInfo___NominalAuxDyn']['evtweight'] * dihiggs_10.fold_0_array['fold_weight']
    
    for klambda in klambda_scan_list:
    
        print('klambda reweight scan round 2: ' + klambda)
        
        if (klambda == '10p0'):
            truth_significances.append(0)
            mdn_significances.append(0)
            mmc_significances.append(0)
            split_significances.append(0)
            split_truth.append(0)
            continue
            
        reweight = reweight_file['reweight_mHH_1p0_to_' + klambda].to_numpy()
        norm = reweight_file['norm' + klambda].value
        
        reweights_by_bin = reweight[0] * norm
        num_bins = len(reweights_by_bin)
        
        new_weights = []
        for i in range(len(dihiggs_10.fold_0_array['universal_true_mhh'])):
            reweight_bin = int((dihiggs_10.fold_0_array['universal_true_mhh'][i] - 200) / 20)
            if ((reweight_bin > -1) and (reweight_bin < num_bins)):
                new_weights.append(original_weights[i] * reweights_by_bin[reweight_bin])
            else:
                new_weights.append(0)
        new_weights = np.array(new_weights)
                
        z, cs_norm = reweight_and_compare(dihiggs_10.fold_0_array['universal_true_mhh'], original_weights, new_weights, 'truth', klambda, k10mode = True)
        truth_significances.append(z)
        
        z, cs_norm = reweight_and_compare(predictions_HH_10, original_weights, new_weights, 'mdn', klambda, k10mode = True, cs_norm = cs_norm)
        mdn_significances.append(z)
        
        z, cs_norm = reweight_and_compare(mhh_mmc_HH_10, original_weights, new_weights, 'mmc', klambda, k10mode = True, cs_norm = cs_norm)
        mmc_significances.append(z)
        
        z1, cs_norm = reweight_and_compare(predictions_HH_10[indices_1_10], original_weights[indices_1_10], new_weights[indices_1_10], 'mdn_good', klambda, k10mode = True, cs_norm = cs_norm)
        z2, cs_norm = reweight_and_compare(predictions_HH_10[indices_2_10], original_weights[indices_2_10], new_weights[indices_2_10], 'mdn_bad', klambda, k10mode = True, cs_norm = cs_norm)
        z = np.sqrt(z1 * z1 + z2 * z2)
        print("MDN Splitting Improvement: " + str(z - mdn_significances[-1]))
        split_significances.append(z)
        
        z1, cs_norm = reweight_and_compare(dihiggs_10.fold_0_array[indices_1_10]['universal_true_mhh'], original_weights[indices_1_10], new_weights[indices_1_10], 'truth_good', klambda, k10mode = True, cs_norm = cs_norm)
        z2, cs_norm = reweight_and_compare(dihiggs_10.fold_0_array[indices_2_10]['universal_true_mhh'], original_weights[indices_2_10], new_weights[indices_2_10], 'truth_bad', klambda, k10mode = True, cs_norm = cs_norm)
        z = np.sqrt(z1 * z1 + z2 * z2)
        print("Truth Splitting Improvement: " + str(z - truth_significances[-1]))
        split_truth.append(z)
        
    z = k_lambda_comparison_plot(dihiggs_10.fold_0_array['universal_true_mhh'], dihiggs_01.fold_0_array['universal_true_mhh'], dihiggs_10.fold_0_array, dihiggs_01.fold_0_array, 'truth_reversed', k10mode = True)
    original_significances.append(z)
    z = k_lambda_comparison_plot(predictions_HH_10, predictions_HH_01, dihiggs_10.fold_0_array, dihiggs_01.fold_0_array, 'mdn_reversed', k10mode = True)
    original_significances.append(z)
    z = k_lambda_comparison_plot(mhh_mmc_HH_10, mhh_mmc_HH_01, dihiggs_10.fold_0_array, dihiggs_01.fold_0_array, 'mmc_reversed', k10mode = True)
    original_significances.append(z)
    z1 = k_lambda_comparison_plot(predictions_HH_10[indices_1_10], predictions_HH_01[indices_1_1], dihiggs_10.fold_0_array[indices_1_10], dihiggs_01.fold_0_array[indices_1_1], 'mdn_good_reversed', k10mode = True)
    z2 = k_lambda_comparison_plot(predictions_HH_10[indices_2_10], predictions_HH_01[indices_2_1], dihiggs_10.fold_0_array[indices_2_10], dihiggs_01.fold_0_array[indices_2_1], 'mdn_bad_reversed', k10mode = True)
    z = np.sqrt(z1 * z1 + z2 * z2)
    original_significances.append(z)
    z1 = k_lambda_comparison_plot(dihiggs_10.fold_0_array[indices_1_10]['universal_true_mhh'], dihiggs_01.fold_0_array['universal_true_mhh'][indices_1_1], dihiggs_10.fold_0_array[indices_1_10], dihiggs_01.fold_0_array[indices_1_1], 'truth_good_reversed', k10mode = True)
    z2 = k_lambda_comparison_plot(dihiggs_10.fold_0_array[indices_2_10]['universal_true_mhh'], dihiggs_01.fold_0_array['universal_true_mhh'][indices_2_1], dihiggs_10.fold_0_array[indices_2_10], dihiggs_01.fold_0_array[indices_2_1], 'truth_bad_reversed', k10mode = True)
    z = np.sqrt(z1 * z1 + z2 * z2)
    original_significances.append(z)
        
    klambda_scan_plot(range(-8, 16), truth_significances, mdn_significances, mmc_significances, split_significances, k10mode = True, bonus_pts = original_significances, split_truth = split_truth)
    #klambda_scan_plot(range(-8, 16), truth_significances, mdn_significances, mmc_significances, split_significances, k10mode = True, bonus_pts = original_significances)
    """
        
        