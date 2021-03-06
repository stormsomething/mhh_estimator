import os
import joblib
import awkward as ak
from argparse import ArgumentParser
from bbtautau.utils import true_mhh, features_table
from bbtautau import log; log = log.getChild('fitter')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--debug', default=False, action='store_true')
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
        # regressor = joblib.load('cache/best_scikit.clf')
        log.info('loading regressor weights')
        regressor = joblib.load('cache/latest_scikit.clf')

    else:
        if args.gridsearch:
            pass
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            regressor = GradientBoostingRegressor(
                n_estimators=2000,
                learning_rate=0.1,
                max_depth=5,
                random_state=0,
                loss='ls',
                verbose=True)

        train_target = ak.flatten(true_mhh(dihiggs_01.fold_0_array))
        train_features = features_table(dihiggs_01.fold_0_array)
        log.info('fitting')
        regressor.fit(train_features, train_target)
        joblib.dump(regressor, 'cache/latest_scikit.clf')


    log.info('plotting')
    features_test_HH_01 = features_table(dihiggs_01.fold_1_array)
    predictions_HH_01 = regressor.predict(features_test_HH_01)
    test_target_HH_01 = ak.flatten(true_mhh(dihiggs_01.fold_1_array))

    features_test_HH_10 = features_table(dihiggs_10.fold_1_array)
    predictions_HH_10 = regressor.predict(features_test_HH_10)
    test_target_HH_10 = ak.flatten(true_mhh(dihiggs_10.fold_1_array))

    from bbtautau.plotting import signal_pred_target_comparison
    signal_pred_target_comparison(
        predictions_HH_10, predictions_HH_01,
        test_target_HH_10, test_target_HH_01,
        dihiggs_10, dihiggs_01)


    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # import mplhep as hep
    # hep.set_style("ATLAS") 
    # mpl.rc('font', **{'family':'serif','serif':['Palatino']})
    # mpl.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
    # mpl.rc('text', usetex=True)    # mpl.rcParams['text.latex.unicode'] = True


    # fig = plt.figure()
    # plt.hist(
    #     predictions_HH_10,
    #     bins=80,
    #     range=(0, 3000),
    #     label=dihiggs_10.title + ' - pred.',
    #     color=dihiggs_10.color,
    #     linestyle='solid',
    #     linewidth=2,
    #     # cumulative=True,
    #     histtype='step')
    # plt.hist(
    #     predictions_HH_01,
    #     bins=80,
    #     range=(0, 3000),
    #     label=dihiggs_01.title + ' - pred.',
    #     color=dihiggs_01.color,
    #     linestyle='solid',
    #     linewidth=2,
    #     # cumulative=True,
    #     histtype='step')
    # plt.hist(
    #     test_target_HH_10,
    #     bins=80,
    #     range=(0, 3000),
    #     label=dihiggs_10.title + ' - truth.',
    #     color=dihiggs_10.color,
    #     linestyle='dashed',
    #     linewidth=2,
    #     # cumulative=True,
    #     histtype='step')
    # plt.hist(
    #     test_target_HH_01,
    #     bins=80,
    #     range=(0, 3000),
    #     label=dihiggs_01.title + ' - truth.',
    #     color=dihiggs_01.color,
    #     linestyle='dashed',
    #     linewidth=2,
    #     # cumulative=True,
    #     histtype='step')
    
    # plt.xlabel(r'$m_{hh}$ [GeV]')
    # plt.ylabel('Raw Simulation Entries')
    # plt.legend(fontsize='small', numpoints=3)
    # fig.savefig('plots/distributions.pdf')
    # plt.close(fig)
    
    # fig = plt.figure()
    # plt.hist(
    #     [
    #         predictions_HH_01 - test_target_HH_01,
    #         predictions_HH_10 - test_target_HH_10,
    #     ],
    #     label=[
    #         dihiggs_01.title,
    #         dihiggs_10.title,
    #     ],
    #     color=[
    #         dihiggs_01.color,
    #         dihiggs_10.color,
    #     ],
    #     bins=160,
    #     range=(-400, 400),
    #     histtype='step')
    # plt.xlabel(r'$m_{hh}$: prediction - truth [GeV]')
    # plt.ylabel('Raw Simulation Entries')
    # plt.legend(fontsize='small', numpoints=3)
    # fig.savefig('plots/deltas.pdf')
    # plt.close(fig)
    
    # fig = plt.figure()
    # plt.hist(
    #     [
    #         predictions_HH_01 / test_target_HH_01,
    #         predictions_HH_10 / test_target_HH_10
    #     ],
    #     label=[
    #         dihiggs_01.title,
    #         dihiggs_10.title,
    #     ],
    #     color=[
    #         dihiggs_01.color,
    #         dihiggs_10.color,
    #     ],
    #     bins=160,
    #     range=(0., 3.),
    #     histtype='step')
    # plt.xlabel(r'$m_{hh}$: prediction / truth [GeV]')
    # plt.ylabel('Raw Simulation Entries')
    # plt.legend(fontsize='small', numpoints=3)
    # fig.savefig('plots/ratios.pdf')
    # plt.close(fig)

    # fig = plt.figure()
    # plt.hist(ak.flatten(dihiggs_01.fold_1_array['taus']['nTracks']))
    # plt.xlabel('nTracks')
    # fig.savefig('plots/nTracks.pdf')
    # plt.close(fig)
    
    # fig = plt.figure()
    # plt.hist(ak.flatten(dihiggs_01.fold_1_array['taus']['isRNNMedium']))
    # plt.xlabel('RNN Medium ID Decision')
    # fig.savefig('plots/rnn_medium_id.pdf')
    # plt.close(fig)


    # fig = plt.figure()
    # plt.hist([
    #     ak.flatten(true_mhh(dihiggs_01.fold_1_array)),
    #     ak.flatten(true_mhh(dihiggs_10.fold_1_array))],
    #          label=[
    #              dihiggs_01.title,
    #              dihiggs_10.title,
    #          ],
    #          color=[
    #              dihiggs_01.color,
    #              dihiggs_10.color,
    #         ],
    #          bins=80,
    #          linewidth=2,
    #          histtype='step')
    # plt.xlabel(r'True $m_{hh}$ [GeV]')
    # plt.legend(fontsize='small', numpoints=3)
    # fig.savefig('plots/true_mhh_signal_comparison.pdf')
    # plt.close(fig)

    # fig = plt.figure()
    # plt.hist([
    #     predictions_HH_01,
    #     predictions_HH_10],
    #          label=[
    #              dihiggs_01.title,
    #              dihiggs_10.title,
    #         ],
    #          color=[
    #              dihiggs_01.color,
    #              dihiggs_10.color,
    #         ],
    #          bins=80,
    #          linewidth=2,
    #          histtype='step')
    # plt.xlabel(r'Estimated $m_{hh}$ [GeV]')
    # plt.legend(fontsize='small', numpoints=3)
    # fig.savefig('plots/estimated_mhh_signal_comparison.pdf')
    # plt.close(fig)

