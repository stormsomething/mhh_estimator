import os
import joblib
import awkward as ak
from argparse import ArgumentParser
from bbtautau.utils import true_mhh, features_table


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

    from bbtautau.database import dihiggs_01, dihiggs_10
    dihiggs_01.process(verbose=True, max_files=max_files)
    dihiggs_10.process(verbose=True, max_files=max_files)

    
    if not args.fit:
        regressor = joblib.load('cache/best_scikit.clf')

    else:
        if args.gridsearch:
            pass
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            regressor = GradientBoostingRegressor(
                n_estimators=5000,
                learning_rate=0.1,
                max_depth=5,
                random_state=0,
                loss='ls')

        train_target = ak.flatten(true_mhh(dihiggs_01.fold_0_array))
        train_features = features_table(dihiggs_01.fold_0_array)
        regressor.fit(train_features, train_target)
        # joblib.dump(best_regressor, 'cache/latest_scikit.clf')



    features_test = features_table(dihiggs_01.fold_1_array)
    predictions = regressor.predict(features_test)
    test_target = ak.flatten(true_mhh(dihiggs_01.fold_1_array))

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['text.usetex'] = True

    fig = plt.figure()
    plt.hist([predictions, test_target], bins=80, label=['prediction', 'truth'], linewidth=2, histtype='step')
    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/distributions.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(predictions - test_target, bins=160, range=(-400, 400))
    plt.xlabel(r'$m_{hh}$: prediction - truth [GeV]')
    fig.savefig('plots/deltas.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(ak.flatten(dihiggs_01.fold_1_array['taus']['nTracks']))
    plt.xlabel('nTracks')
    fig.savefig('plots/nTracks.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(ak.flatten(dihiggs_01.fold_1_array['taus']['isRNNMedium']))
    plt.xlabel('RNN Medium ID Decision')
    fig.savefig('plots/rnn_medium_id.pdf')
    plt.close(fig)


    fig = plt.figure()
    plt.hist([
        ak.flatten(true_mhh(dihiggs_01.fold_1_array)),
        ak.flatten(true_mhh(dihiggs_10.fold_1_array))],
             label=[
                 dihiggs_01.title,
                 dihiggs_10.title,
             ],
             color=[
                 dihiggs_01.color,
                 dihiggs_10.color,
            ],
             bins=80,
             linewidth=2,
             histtype='step')
    plt.xlabel(r'True $m_{hh}$ [GeV]')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/true_mhh_signal_comparison.pdf')
    plt.close(fig)

    features_test_01 = features_test
    features_test_10 = features_table(dihiggs_10.fold_1_array)
    predictions_10 = regressor.predict(features_test_10)
    fig = plt.figure()
    plt.hist([
        predictions,
        predictions_10],
             label=[
                 dihiggs_01.title,
                 dihiggs_10.title,
            ],
             color=[
                 dihiggs_01.color,
                 dihiggs_10.color,
            ],
             bins=80,
             linewidth=2,
             histtype='step')
    plt.xlabel(r'Estimated $m_{hh}$ [GeV]')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/estimated_mhh_signal_comparison.pdf')
    plt.close(fig)

