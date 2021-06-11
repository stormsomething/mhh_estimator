import os
import joblib
import awkward as ak
import numpy as np
from keras.models import load_model
from argparse import ArgumentParser
from bbtautau.utils import true_mhh, features_table
from bbtautau import log; log = log.getChild(__file__)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--scikit', default='latest_scikit.clf')
    parser.add_argument('--keras', default='latest_keras.h5')
    parser.add_argument('--include-mmc', default=False, action='store_true')
    args = parser.parse_args()

    # use all root files by default
    max_files = None
    if args.debug:
        max_files = 1

    log.info('loading samples ..')
    from bbtautau.database import dihiggs_01
    dihiggs_01.process(verbose=True, is_signal=True, max_files=max_files)
    log.info('..done')
    

    _total_mhh = true_mhh(dihiggs_01.ak_array)
    _total_w = dihiggs_01.ak_array['EventInfo___NominalAuxDyn']['evtweight']

    _train_mhh = true_mhh(dihiggs_01.fold_0_array)
    _train_w = ak.to_numpy(dihiggs_01.fold_0_array['EventInfo___NominalAuxDyn']['evtweight'])
    _train_w *= ak.to_numpy(dihiggs_01.fold_0_array['fold_weight'])

    _test_mhh = true_mhh(dihiggs_01.fold_1_array)
    _test_w = ak.to_numpy(dihiggs_01.fold_1_array['EventInfo___NominalAuxDyn']['evtweight'])
    _test_w *= ak.to_numpy(dihiggs_01.fold_1_array['fold_weight'])

    print (_total_w)
    print (_train_w)
    print (_test_w)
    
    log.info('plotting')

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style("ATLAS") 

    mpl.rc('font', **{'family':'serif','serif':['Palatino']})
    mpl.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
    mpl.rc('text', usetex=True)    # mpl.rcParams['text.latex.unicode'] = True
    fig = plt.figure()
    plt.hist(
        _total_mhh,
        weights=_total_w,
        label=dihiggs_01.title + '(total)',
        color='black',
        bins=80,
        range=(0, 1500),
        linewidth=2,
        histtype='step')
    plt.hist(
        _train_mhh,
        weights=_train_w,
        label=dihiggs_01.title + '(train)',
        color='red',
        bins=80,
        range=(0, 1500),
        linewidth=2,
        histtype='step')
    plt.hist(
        _test_mhh,
        weights=_test_w,
        label=dihiggs_01.title + '(test)',
        color='blue',
        bins=80,
        range=(0, 1500),
        linewidth=2,
        histtype='step')


    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/mhh_{}_train_test_with_weights.pdf'.format(dihiggs_01.name))
    plt.close(fig)

