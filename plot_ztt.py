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
    from bbtautau.database import ztautau
    ztautau.process(verbose=True, is_signal=False, max_files=max_files, use_cache=args.use_cache)
    log.info('..done')
    
    log.info('loading regressor weights')
    scikit = joblib.load(os.path.join('cache', args.scikit))
    keras = load_model(os.path.join('cache', args.keras))

    from bbtautau.utils import features_table, true_mhh
    features_ztautau = features_table(ztautau.ak_array)

    scikit_ztautau = scikit.predict(features_ztautau)
    keras_ztautau = keras.predict(features_ztautau)
    keras_ztautau = np.reshape(
            keras_ztautau, (keras_ztautau.shape[0], ))

    if args.include_mmc:
        from bbtautau.mmc import mmc
        mmc_ztautau, mmc_mhh_ztautau = mmc(ztautau.ak_array)
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
        scikit_ztautau,
        weights=ztautau.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
        label=ztautau.title + '(scikit)',
        color='red',
        bins=80,
        range=(0, 1500),
        linewidth=2,
        histtype='step')
    plt.hist(
        keras_ztautau,
        weights=ztautau.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
        label=ztautau.title + '(keras)',
        color='blue',
        bins=80,
        range=(0, 1500),
        linewidth=2,
        histtype='step')
    if args.include_mmc:
        plt.hist(
            mmc_mhh_ztautau,
            weights=ztautau.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
            bins=80,
            range=(0, 1500),
            label=ztautau.title + '(mmc)',
            color='green',
            linestyle='solid',
            linewidth=2,
        histtype='step')
        


    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/mhh_{}.pdf'.format(ztautau.name))
    plt.close(fig)

