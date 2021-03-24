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
    
    log.info('plotting')
    test_target_HH_01 = ak.flatten(true_mhh(dihiggs_01.fold_1_array))

    test_target_HH_10 = ak.flatten(true_mhh(dihiggs_10.fold_1_array))

    from bbtautau.plotting import signal_pred_target_comparison, signal_features
    from bbtautau.mmc import mmc

    mmc_01, mhh_01 = mmc(dihiggs_01.fold_1_array)
    mmc_10, mhh_10 = mmc(dihiggs_10.fold_1_array)
    signal_pred_target_comparison(
        mhh_10, mhh_01,
        test_target_HH_10, test_target_HH_01,
        dihiggs_10, dihiggs_01,
        regressor='MMC')
