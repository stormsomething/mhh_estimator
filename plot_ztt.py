import os
import joblib
import awkward as ak
import numpy as np
from keras.models import load_model
from argparse import ArgumentParser
from bbtautau.utils import true_mhh, features_table
from bbtautau import log; log = log.getChild('plot_ztt')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--scikit', default='latest_scikit.clf')
    parser.add_argument('--keras', default='latest_keras.h5')
    parser.add_argument('--include-mmc', default=False, action='store_true')
    parser.add_argument('--use-cache', default=False, action='store_true')
    args = parser.parse_args()

    # use all root files by default
    max_files = None
    if args.debug:
        max_files = 1

    log.info('loading samples ..')
    from bbtautau.database import ztautau, dihiggs_01, dihiggs_10
    ztautau.process(verbose=True, is_signal=False, max_files=max_files, use_cache=args.use_cache)
    dihiggs_01.process(verbose=True, max_files=max_files, use_cache=args.use_cache)
    dihiggs_10.process(verbose=True, max_files=max_files, use_cache=args.use_cache)
    log.info('..done')

    log.info('loading regressor weights')
    scikit = joblib.load(os.path.join('cache', args.scikit))
    keras = load_model(os.path.join('cache', args.keras))

    from bbtautau.utils import features_table, true_mhh
    features_ztautau = features_table(ztautau.ak_array)
    features_HH_01 = features_table(dihiggs_01.ak_array)
    features_HH_10 = features_table(dihiggs_10.ak_array)

    scikit_ztautau = scikit.predict(features_ztautau)
    scikit_dihiggs_01 = scikit.predict(features_HH_01)
    scikit_dihiggs_10 = scikit.predict(features_HH_10)
    keras_ztautau = keras.predict(features_ztautau)
    keras_dihiggs_01 = keras.predict(features_HH_01)
    keras_dihiggs_10 = keras.predict(features_HH_10)
    keras_ztautau = np.reshape(
            keras_ztautau, (keras_ztautau.shape[0], ))
    keras_dihiggs_01 = np.reshape(
            keras_dihiggs_01, (keras_dihiggs_01.shape[0], ))
    keras_dihiggs_10 = np.reshape(
            keras_dihiggs_10, (keras_dihiggs_10.shape[0], ))

    if args.include_mmc:
        from bbtautau.mmc import mmc
        mmc_ztautau, mmc_mhh_ztautau = mmc(ztautau.ak_array)
        mmc_dihiggs_01, mmc_mhh_dihiggs_01 = mmc(dihiggs_01.ak_array)
        mmc_mhh_dihiggs_01 = mmc_mhh_dihiggs_01
        mmc_dihiggs_10, mmc_mhh_dihiggs_10 = mmc(dihiggs_10.ak_array)
        mmc_mhh_dihiggs_10 = mmc_mhh_dihiggs_10
    log.info('plotting')

    from bbtautau.plotting import signal_ztt_distributions_overlay, signal_ztt_pt_overlay
    signal_ztt_distributions_overlay(ztautau, dihiggs_01, dihiggs_10, scikit_ztautau, scikit_dihiggs_01, scikit_dihiggs_10, keras_ztautau, keras_dihiggs_01, keras_dihiggs_10, [mmc_ztautau, mmc_mhh_ztautau] if args.include_mmc else None, [mmc_dihiggs_01, mmc_mhh_dihiggs_01] if args.include_mmc else None, [mmc_dihiggs_10, mmc_mhh_dihiggs_10] if args.include_mmc else None)
    signal_ztt_pt_overlay(ztautau, dihiggs_01, dihiggs_10)
