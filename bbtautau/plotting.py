import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import scipy.stats as sc
import numpy as np
from math import sqrt
from bbtautau.utils import gauss_fit_calculator
from sklearn.metrics import mean_squared_error, auc
hep.set_style("ATLAS")

mpl.rc('font', **{'family':'serif','serif':['Palatino']})
mpl.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)    # mpl.rcParams['text.latex.unicode'] = True

from . import log; log = log.getChild(__name__)


def rnn_mmc_comparison(predictions_rnn, test_target, ak_array, ak_array_fold_1_array, label, regressor, predictions_mmc = None):

    eff_tot = []
    fig = plt.figure()
    weights = ak_array_fold_1_array['EventInfo___NominalAuxDyn']['evtweight']*ak_array_fold_1_array['fold_weight']

    rms_rnn = sqrt(mean_squared_error(predictions_rnn, test_target))
    if predictions_mmc is not None:
        rms_mmc = sqrt(mean_squared_error(predictions_mmc, test_target))

    (n_rnn, bins_rnn, patches_rnn) = plt.hist(
        predictions_rnn,
        bins=80,
        weights=weights,
        range=(0,1500),
        #label=ak_array.title + '- RNN. Raw RMS: ' + str(round(rms_rnn, 4)) + '.',
        label=ak_array.title + '- DNN. Raw RMS: ' + str(round(rms_rnn, 4)) + '.',
        color=ak_array.color,
        linestyle='solid',
        linewidth=2,
        histtype='step')

    eff_tot.append(calculate_eff(n_rnn))

    plt.hist(
        test_target,
        bins=80,
        weights= weights,
        range=(0,1500),
        label=ak_array.title + '- truth.',
        color=ak_array.color,
        linestyle='dashed',
        linewidth=2,
        histtype='step')

    if predictions_mmc is not None:
        (n_mmc, bins_mmc, patches_mmc) = plt.hist(
            predictions_mmc,
            bins=80,
            weights= weights,
            range=(0,1500),
            label = ak_array.title + '- MMC. Raw RMS: ' + str(round(rms_mmc, 4)) + '.',
            #label = ak_array.title + '- Original RNN. Raw RMS: ' + str(round(rms_mmc, 4)) + '.',
            color='purple',
            linestyle='solid',
            linewidth=2,
            histtype='step')

        eff_tot.append(calculate_eff(n_mmc))

    plt.xlabel(r'$m_{HH}$ [GeV]')
    plt.ylabel('Events')
    plt.ylim(bottom=0, top=max(n_rnn)*1.4)
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_distributions_{}.pdf'.format(regressor))
    plt.close(fig)

    fig = plt.figure()

    (n_true, bins_true, patches_true) = plt.hist(
        test_target,
        bins=80,
        weights= weights,
        range=(0,1500),
        label=ak_array.title,
        color=ak_array.color,
        linestyle='dashed',
        linewidth=2,
        histtype='step')

    eff_true = calculate_eff(n_true)

    plt.xlabel(r'True $m_{HH}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_distributions_truthonly.pdf')
    plt.close(fig)

    fig = plt.figure()

    ratio_rnn = np.array(predictions_rnn / test_target)
    avg_ratio_rnn = np.average(ratio_rnn)
    target_ratio_rnn = [1.0] * len(ratio_rnn)
    rms_ratio_rnn = sqrt(mean_squared_error(ratio_rnn, target_ratio_rnn))

    if predictions_mmc is not None:
        ratio_mmc = np.array(predictions_mmc / test_target)
        avg_ratio_mmc = np.average(ratio_mmc)
        target_ratio_mmc = [1.0] * len(ratio_mmc)
        rms_ratio_mmc = sqrt(mean_squared_error(ratio_mmc, target_ratio_mmc))

    (n_rat_rnn, bins_rat_rnn, patches_rat_rnn) = plt.hist(
        ratio_rnn,
        #label= ak_array.title + '- RNN. Raw Mean: ' + str(round(avg_ratio_rnn, 4)) + '. Raw RMS: ' + str(round(rms_ratio_rnn, 4)) + '.',
        label= ak_array.title + '- DNN. Raw Mean: ' + str(round(avg_ratio_rnn, 4)) + '. Raw RMS: ' + str(round(rms_ratio_rnn, 4)) + '.',
        color=ak_array.color,
        weights= weights,
        bins=160,
        range=(0., 3.),
        linewidth=2,
        histtype='step')

    gauss_fit_calculator(n_rat_rnn, bins_rat_rnn, label, 'RNN', new_label = 1)

    if predictions_mmc is not None:
        (n_rat_mmc, bins_rat_mmc, patches_rat_mmc) = plt.hist(
            ratio_mmc,
            label= ak_array.title + '- MMC. Raw Mean: ' + str(round(avg_ratio_mmc, 4)) + '. Raw RMS: ' + str(round(rms_ratio_mmc, 4)) + '.',
            #label= ak_array.title + '- Original RNN. Raw Mean: ' + str(round(avg_ratio_mmc, 4)) + '. Raw RMS: ' + str(round(rms_ratio_mmc, 4)) + '.',
            color='purple',
            weights= weights,
            bins=160,
            range=(0., 3.),
            linewidth=2,
            histtype='step')

        gauss_fit_calculator(n_rat_mmc, bins_rat_mmc, label, 'MMC')
        #gauss_fit_calculator(n_rat_mmc, bins_rat_mmc, label, 'RNN', new_label = 2)

    plt.xlabel(r'$m_{HH}$: prediction / truth')
    plt.ylabel('Events')
    plt.ylim(bottom=0, top=max(n_rat_rnn)*1.4)
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_ratios_{}.pdf'.format(regressor))
    plt.close(fig)

    return eff_tot, eff_true, n_rnn, n_mmc, n_true


def avg_mhh_calculation(ak_array, truth, keras, mmc):

    avg_mu = ak_array['EventInfo___NominalAuxDyn']['CorrectedAndScaledAvgMu']

    avg_mhh_truth = []
    mhh_truth_sem = []

    avg_mhh_keras = []
    mhh_keras_sem = []

    if mmc is not None:
        avg_mhh_mmc = []
        mhh_mmc_sem = []

    bins = [[0,20], [20,30], [30,40], [40,50], [50,60], [60,85]]

    i = 0
    while i in range(len(bins)):
        mod_truth = []
        mod_keras = []
        if mmc is not None:
            mod_mmc = []
        for n in range(len(truth)):
            if avg_mu[n] >= bins[i][0] and avg_mu[n] < bins[i][1]:
                mod_truth.append(truth[n])
                mod_keras.append(keras[n])
                if mmc is not None:
                    mod_mmc.append(mmc[n])

        avg_truth = sum(mod_truth) / len(mod_truth)
        sem_truth = sc.sem(mod_truth)
        avg_mhh_truth.append(avg_truth)
        mhh_truth_sem.append(sem_truth)

        avg_keras = sum(mod_keras) / len(mod_keras)
        sem_keras = sc.sem(mod_keras)
        avg_mhh_keras.append(avg_keras)
        mhh_keras_sem.append(sem_keras)

        if mmc is not None:
            avg_mmc = sum(mod_mmc) / len(mod_mmc)
            sem_mmc = sc.sem(mod_mmc)
            avg_mhh_mmc.append(avg_mmc)
            mhh_mmc_sem.append(sem_mmc)

        i = i + 1

    avg_mhh = [avg_mhh_truth, mhh_truth_sem, avg_mhh_keras, mhh_keras_sem]
    if mmc is not None:
        avg_mhh.append(avg_mhh_mmc)
        avg_mhh.append(mhh_mmc_sem)

    return avg_mhh


def avg_mhh_plot(avg_mhh, name, sample):

    x = [0, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 85]
    y_truth = []
    y_keras = []
    if len(avg_mhh) == 6:
        y_mmc = []
    for i in range(int(len(x)/2)):
        y_truth.append(avg_mhh[0][i])
        y_truth.append(avg_mhh[0][i])
        y_keras.append(avg_mhh[2][i])
        y_keras.append(avg_mhh[2][i])
        if len(avg_mhh) == 6:
            y_mmc.append(avg_mhh[4][i])
            y_mmc.append(avg_mhh[4][i])
    fig = plt.figure()
    plt.plot(x, y_keras, color = 'red', label = 'keras')
    plt.plot(x, y_truth, color = 'green', label = 'truth')
    if len(avg_mhh) == 6:
        plt.plot(x, y_mmc, color = 'purple', label = 'mmc')
    plot_error_bars(avg_mhh)
    plt.xlim(0, 85)
    plt.xlabel('Average number of pp interactions per bunch crossing')
    plt.ylabel(r'$<m_{HH}>$ in each bin +/- SEM [GeV]')
    plt.legend(title = sample.title)
    fig.savefig('plots/' + name + '.pdf')
    plt.close(fig)


def plot_error_bars(avg_mhh):

    x = [10, 25, 35, 45, 55, 72.5]
    y_truth = avg_mhh[0]
    truth_err = avg_mhh[1]
    y_keras = avg_mhh[2]
    keras_err = avg_mhh[3]
    if len(avg_mhh) == 6:
        y_mmc = avg_mhh[4]
        mmc_err = avg_mhh[5]
    plt.errorbar(x, y_truth, yerr = truth_err, fmt = 'none', ecolor = 'green', capsize = 6)
    plt.errorbar(x, y_keras, yerr = keras_err, fmt = 'none', ecolor = 'red', capsize = 6)
    if len(avg_mhh) == 6:
        plt.errorbar(x, y_mmc, yerr = mmc_err, fmt = 'none', ecolor = 'purple', capsize = 6)


def calculate_eff(n):

    tot = 0
    for N in range(len(n)):
        tot = tot + n[N]

    N = 79
    eff = []
    subtot = 0
    while N in range(len(n)):
        subtot = subtot + abs(n[N])
        eff.append(subtot/tot)
        N = N - 1
    eff.reverse()

    return eff


def roc_plot_rnn_mmc(eff, eff_true, name_1, name_2):

    auc_rnn = auc(eff[0], eff[2])
    auc_mmc = auc(eff[1], eff[3])
    auc_true = auc(eff_true[0], eff_true[1])

    fig = plt.figure()
    #plt.plot(eff[0], eff[2], color = 'red', label = 'RNN. AUC = ' + str(auc_rnn) + '.')
    plt.plot(eff[0], eff[2], color = 'red', label = 'New NN. AUC = ' + str(auc_rnn) + '.')
    #plt.plot(eff[1], eff[3], color = 'green', label = 'MMC. AUC = ' + str(auc_mmc) + '.')
    plt.plot(eff[1], eff[3], color = 'green', label = 'Original RNN. AUC = ' + str(auc_mmc) + '.')
    plt.plot(eff_true[0], eff_true[1], color = 'purple', label = 'Truth. AUC = ' + str(auc_true) + '.')
    plt.xlabel(name_1 + ' Efficiency')
    plt.ylabel(name_2 + ' Efficiency')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend()
    if (name_1 == r'$\kappa_{\lambda}$ = 1'):
        title_1 = 'HH_01'
    if (name_2 == r'$\kappa_{\lambda}$ = 1'):
        title_2 = 'HH_01'
    if (name_1 == r'$\kappa_{\lambda}$ = 10'):
        title_1 = 'HH_10'
    if (name_2 == r'$\kappa_{\lambda}$ = 10'):
        title_2 = 'HH_10'
    if (name_1 == r'$Z\to\tau\tau$ + jets'):
        title_1 = 'ztt'
    if (name_2 == r'$Z\to\tau\tau$ + jets'):
        title_2 = 'ztt'
    if (name_1 == 'Top Quark'):
        title_1 = 'ttbar'
    if (name_2 == 'Top Quark'):
        title_2 = 'ttbar'
    fig.savefig('plots/' + title_1 + '_' + title_2 + '_mmc_rnn_roc.pdf')
    plt.close(fig)


def nn_history(history, metric='loss'):

    fig = plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig.axes[0].set_yscale('log')
    fig.savefig('plots/nn_model_{}.pdf'.format(metric))
    plt.close(fig)




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The below functions are not currently in use.


def signal_pred_target_comparison(
        predictions_HH_10,
        predictions_HH_01,
        test_target_HH_10,
        test_target_HH_01,
        dihiggs_10,
        dihiggs_01,
        label,
        regressor='keras'):

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    eff_tot = []
    log.info('plotting distributions')

    weights_HH_10 = dihiggs_10.fold_1_array['EventInfo___NominalAuxDyn']['evtweight']
    weights_HH_01 = dihiggs_01.fold_1_array['EventInfo___NominalAuxDyn']['evtweight']

    rms_HH_10 = sqrt(mean_squared_error(predictions_HH_10, test_target_HH_10))
    rms_HH_01 = sqrt(mean_squared_error(predictions_HH_01, test_target_HH_01))

    fig = plt.figure()

    (n_HH_10, bins_HH_10, patches_HH_10) = plt.hist(
        predictions_HH_10,
        bins=80,
        weights=weights_HH_10,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_10.title + ' - pred. RMS: ' + str(round(rms_HH_10, 4)),
        color=dihiggs_10.color,
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    eff_tot.append(calculate_eff(n_HH_10))

    plt.hist(
        test_target_HH_10,
        bins=80,
        weights=weights_HH_10,
        # bins=15,
        range=(0, 1500),
        label=dihiggs_10.title + ' - truth.',
        color=dihiggs_10.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    (n_HH_01, bins_HH_01, patches_HH_01) = plt.hist(
        predictions_HH_01,
        bins=80,
        weights=weights_HH_01,
        # bins=15,
        range=(0, 1500),
        label=dihiggs_01.title + ' - pred. RMS: ' + str(round(rms_HH_01, 4)),
        color=dihiggs_01.color,
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    eff_tot.append(calculate_eff(n_HH_01))

    plt.hist(
        test_target_HH_01,
        bins=80,
        weights=weights_HH_01,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_01.title + ' - truth.',
        color=dihiggs_01.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3, title=regressor)
    plt.ylim(bottom=0)
    fig.savefig('plots/' + str(label) + '_distributions_{}.pdf'.format(regressor))
    plt.close(fig)

    fig = plt.figure()
    plt.hist(
        test_target_HH_10,
        bins=80,
        weights=weights_HH_10,
        range=(0, 1500),
        label=dihiggs_10.title,
        color=dihiggs_10.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    plt.hist(
        test_target_HH_01,
        bins=80,
        weights=weights_HH_01,
        range=(0, 1500),
        label=dihiggs_01.title,
        color=dihiggs_01.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    plt.xlabel(r'True $m_{hh}$ [GeV]')
    plt.ylabel('Events')
    plt.ylim(bottom=0)
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_distributions_truthonly.pdf')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(
        [predictions_HH_01 - test_target_HH_01, predictions_HH_10 - test_target_HH_10,],
        label=[dihiggs_01.title, dihiggs_10.title,],
        color=[dihiggs_01.color, dihiggs_10.color,],
        bins=160,
        weights = [weights_HH_01, weights_HH_10,],
        range=(-400, 400),
        linewidth=2,
        histtype='step')

    plt.xlabel(r'$m_{hh}$: prediction - truth [GeV]')
    plt.ylabel('Events')
    plt.ylim(bottom=0)
    plt.legend(fontsize='small', numpoints=3, title=regressor)
    fig.savefig('plots/' + str(label) + '_deltas_{}.pdf'.format(regressor))
    plt.close(fig)

    fig = plt.figure()

    ratio_HH_01 = predictions_HH_01 / test_target_HH_01
    ratio_HH_10 = predictions_HH_10 / test_target_HH_10
    avg_ratio_HH_01 = sum(ratio_HH_01) / len(ratio_HH_01)
    avg_ratio_HH_10 = sum(ratio_HH_10) / len(ratio_HH_10)
    pred_ratio_HH_01 = [1.0] * len(ratio_HH_01)
    pred_ratio_HH_10 = [1.0] * len(ratio_HH_10)

    rms_ratio_HH_01 = sqrt(mean_squared_error(ratio_HH_01, pred_ratio_HH_01))
    rms_ratio_HH_10 = sqrt(mean_squared_error(ratio_HH_10, pred_ratio_HH_10))

    plt.hist(
        [ratio_HH_01, ratio_HH_10,],
        label=[
            r'$\kappa_{\lambda}$ = 1. Mean: ' + str(round(avg_ratio_HH_01, 4)) + '. RMS: ' + str(round(rms_ratio_HH_01, 4)) + '.',
            r'$\kappa_{\lambda}$ = 10. Mean: ' + str(round(avg_ratio_HH_10, 4)) + '. RMS: ' + str(round(rms_ratio_HH_10, 4)) + '.',
        ],
        color=[dihiggs_01.color, dihiggs_10.color,],
        weights = [weights_HH_01, weights_HH_10,],
        bins=160,
        range=(0., 3.),
        linewidth=2,
         histtype='step')
    plt.xlabel(r'$m_{hh}$: prediction / truth [GeV]')
    plt.ylabel('Events')
    plt.ylim(bottom=0)
    plt.legend(fontsize='small', numpoints=3, title=regressor)
    fig.savefig('plots/' + str(label) + '_ratios_{}.pdf'.format(regressor))
    plt.close(fig)

    return eff_tot


def signal_features(dihiggs_01, dihiggs_10, label):

    weights_HH_10 = dihiggs_10.fold_1_array['EventInfo___NominalAuxDyn']['evtweight']
    weights_HH_01 = dihiggs_01.fold_1_array['EventInfo___NominalAuxDyn']['evtweight']

    fig = plt.figure()
    plt.hist(
        ak.flatten(dihiggs_01.fold_1_array['taus']['pt'] / 1000.),
        label=dihiggs_01.title,
        color=dihiggs_01.color,
        weights=weights_HH_01,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')

    plt.hist(
        ak.flatten(dihiggs_10.fold_1_array['taus']['pt'] / 1000.),
        label=dihiggs_10.title,
        color=dihiggs_10.color,
        weights=weights_HH_10,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')

    plt.xlabel(r'Hadronic taus $p_{T}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_hadronic_taus_pt.pdf')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(
        ak.flatten(dihiggs_01.fold_1_array['bjets']['pt'] / 1000.),
        label=dihiggs_01.title,
        color=dihiggs_01.color,
        weights = weights_HH_01,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')

    plt.hist(
        ak.flatten(dihiggs_10.fold_1_array['bjets']['pt'] / 1000.),
        label=dihiggs_10.title,
        color=dihiggs_10.color,
        weights = weights_HH_10,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')

    plt.xlabel(r'b-jets $p_{T}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_hadronic_bjets_pt.pdf')
    plt.close(fig)


def ztautau_pred_target_comparison(predictions_ztautau, test_target_ztautau, ztautau, label, regressor, predictions_mmc = None):

    from sklearn.metrics import mean_squared_error
    from math import sqrt

    eff_tot = []
    fig = plt.figure()

    weights_ztt = ztautau.fold_1_array['EventInfo___NominalAuxDyn']['evtweight']

    rms_ztautau = sqrt(mean_squared_error(predictions_ztautau, test_target_ztautau))
    if predictions_mmc is not None:
        rms_mmc = sqrt(mean_squared_error(predictions_mmc, test_target_ztautau))

    (n_ztt_rnn, bins_ztt_rnn, patches_ztt_rnn) = plt.hist(
        predictions_ztautau,
        bins=80,
        weights=weights_ztt,
        range=(0, 1500),
        label=ztautau.title + '- pred (RNN). RMS: ' + str(round(rms_ztautau, 4)),
        color='green',
        linestyle='solid',
        linewidth=2,
        histtype='step')

    eff_tot.append(calculate_eff(n_ztt_rnn))

    plt.hist(
        test_target_ztautau,
        bins=80,
        weights= weights_ztt,
        range=(0,1500),
        label=ztautau.title + '- truth.',
        color='green',
        linestyle='dashed',
        linewidth=2,
        histtype='step')

    if predictions_mmc is not None:
        (n_ztt_mmc, bins_ztt_mmc, patches_ztt_mmc) = plt.hist(
            predictions_mmc,
            bins=80,
            weights= weights_ztt,
            range=(0,1500),
            label = ztautau.title + '- pred (MMC). RMS: ' + str(round(rms_mmc, 4)),
            color='purple',
            linestyle='solid',
            linewidth=2,
            histtype='step')

        eff_tot.append(calculate_eff(n_ztt_mmc))

    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.ylabel('Events')
    plt.ylim(bottom=0)
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_distributions_{}.pdf'.format(regressor))
    plt.close(fig)

    fig = plt.figure()

    plt.hist(
        test_target_ztautau,
        bins=80,
        weights= weights_ztt,
        range=(0,1500),
        label=ztautau.title,
        color='green',
        linestyle='dashed',
        linewidth=2,
        histtype='step')

    plt.xlabel(r'True $m_{hh}$ [GeV]')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_distributions_truthonly.pdf')
    plt.close(fig)

    fig = plt.figure()
    ratio_ztautau = predictions_ztautau / test_target_ztautau
    avg_ratio_ztautau = sum(ratio_ztautau) / len(ratio_ztautau)

    from sklearn.metrics import mean_squared_error
    pred_ratio_ztautau = []
    for i in range(len(ratio_ztautau)):
        pred_ratio_ztautau.append(1.0)

    from math import sqrt
    rms_ratio_ztautau = sqrt(mean_squared_error(ratio_ztautau, pred_ratio_ztautau))

    if predictions_mmc is not None:
        ratio_mmc = predictions_mmc / test_target_ztautau
        avg_ratio_mmc = sum(ratio_mmc) / len(ratio_mmc)

        pred_ratio_mmc = []
        for i in range(len(ratio_mmc)):
            pred_ratio_mmc.append(1.0)

        rms_ratio_mmc = sqrt(mean_squared_error(ratio_mmc, pred_ratio_mmc))

    plt.hist(
        ratio_ztautau,
        label= ztautau.title + ' (RNN). Mean: ' + str(round(avg_ratio_ztautau, 4)) + '. RMS: ' + str(round(rms_ratio_ztautau, 4)) + '.',
        color='green',
        weights= weights_ztt,
        bins=160,
        range=(0., 3.),
        linewidth=2,
        histtype='step')

    if predictions_mmc is not None:
        plt.hist(
            ratio_mmc,
            label= ztautau.title + ' (MMC). Mean: ' + str(round(avg_ratio_mmc, 4)) + '. RMS: ' + str(round(rms_ratio_mmc, 4)) + '.',
            color='purple',
            weights= weights_ztt,
            bins=160,
            range=(0., 3.),
            linewidth=2,
            histtype='step')


    plt.xlabel(r'$m_{hh}$: prediction / truth [GeV]')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_ratios_{}.pdf'.format(regressor))
    plt.close(fig)

    return eff_tot


def compare_ml(
        scikit,
        keras,
        truth,
        sample,
        mmc=None):

    log.info('plotting distributions')

    eff_tot = []
    fig = plt.figure()

    (n_s, bins_s, patches_s) = plt.hist(
        scikit,
        bins=80,
        range=(0, 1500),
        label='scikit',
        color='red',
        linestyle='solid',
        linewidth=2,
        histtype='step')

    eff_tot.append(calculate_eff(n_s))

    (n_k, bins_k, patches_k) = plt.hist(
        keras,
        bins=80,
        range=(0, 1500),
        label='keras',
        color='blue',
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    eff_tot.append(calculate_eff(n_k))

    (n_t, bins_t, patches_t) = plt.hist(
        truth,
        bins=80,
        range=(0, 1500),
        label='truth.',
        color='black',
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    eff_tot.append(calculate_eff(n_t))

    if type(mmc) != type(None):
        (n_m, bins_m, patches_m) = plt.hist(
            mmc,
            bins=80,
            range=(0, 1500),
            label='MMC',
            color='green',
            linestyle='solid',
            linewidth=2,
        histtype='step')

        eff_tot.append(calculate_eff(n_m))

    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3, title=sample.title)
    fig.savefig('plots/distributions_allmethods_{}.pdf'.format(sample.name))
    plt.close(fig)

    _hists = [
            scikit / truth,
            keras / truth,
        ]
    _labels = [
        'scikit BRT',
        'keras NN',
        ]
    _colors = [
        'red',
        'blue',
        ]
    if type(mmc) != type(None):
        _hists += [mmc / truth]
        _labels += ['MMC']
        _colors += ['green']

    fig = plt.figure()
    plt.hist(
        _hists,
        label=_labels,
        color=_colors,
        bins=160,
        range=(0., 3.),
        linewidth=2,
        histtype='step')
    plt.xlabel(r'$m_{hh}$: prediction / truth [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3, title=sample.title)
    fig.savefig('plots/ratios_allmethods_{}.pdf'.format(sample.name))
    plt.close(fig)

    return eff_tot


def rms_mhh_plot(rms_mhh, name, sample, num):

    x = [0, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 85]
    y_truth = []
    y_scikit = []
    y_keras = []
    if len(rms_mhh) == 4:
        y_mmc = []
    for i in range(int(len(x)/2)):
        y_truth.append(rms_mhh[0][i])
        y_truth.append(rms_mhh[0][i])
        y_scikit.append(rms_mhh[1][i])
        y_scikit.append(rms_mhh[1][i])
        y_keras.append(rms_mhh[2][i])
        y_keras.append(rms_mhh[2][i])
        if len(rms_mhh) == 4:
            y_mmc.append(rms_mhh[3][i])
            y_mmc.append(rms_mhh[3][i])
    fig = plt.figure()
    plt.plot(x, y_scikit, color = 'blue', label = 'scikit')
    plt.plot(x, y_keras, color = 'red', label = 'keras')
    plt.plot(x, y_truth, color = 'green', label = 'truth')
    if len(rms_mhh) == 4:
        plt.plot(x, y_mmc, color = 'purple', label = 'mmc')
#    if len(avg_mhh) == 6:
#        if num == 1:
#            plt.ylim(485, 545)
#        if num == 10:
#            plt.ylim(370, 410)
#    if len(avg_mhh) == 8:
#        if num == 1:
#            plt.ylim(400, 540)
#        if num == 10:
#            plt.ylim(300, 410)
#    plot_error_bars(rms_mhh)
    plt.xlim(0, 85)
    plt.xlabel('Average number of pp interactions per bunch crossing')
    plt.ylabel(r'RMS($m_{hh}$) in each bin [GeV]')
    plt.legend(title = sample.title)
    fig.savefig('plots/' + name + '.pdf')
    plt.close(fig)


def signal_ztt_distributions_overlay(ztautau, dihiggs_01, dihiggs_10, scikit_ztautau, scikit_dihiggs_01, scikit_dihiggs_10, keras_ztautau, keras_dihiggs_01, keras_dihiggs_10, mmc_ztautau=None, mmc_dihiggs_01=None, mmc_dihiggs_10=None):

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
    if mmc_ztautau is not None:
        plt.hist(
            mmc_ztautau[1],
            weights=ztautau.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
            bins=80,
            range=(0, 1500),
            label=ztautau.title + '(mmc)',
            color='green',
            linestyle='solid',
            linewidth=2,
        histtype='step')

    plt.hist(
        scikit_dihiggs_01,
        weights=dihiggs_01.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
        bins=80,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_01.title + '(scikit)',
        color='purple',
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        scikit_dihiggs_10,
        weights=dihiggs_10.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
        bins=80,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_10.title + '(scikit)',
        color='orange',
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        keras_dihiggs_01,
        weights=dihiggs_01.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
        bins=80,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_01.title + '(keras)',
        color='pink',
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        keras_dihiggs_10,
        weights=dihiggs_10.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
        bins=80,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_10.title + '(keras)',
        color='black',
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    if mmc_dihiggs_01 is not None:
        plt.hist(
            mmc_dihiggs_01[1],
            weights=dihiggs_01.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
            bins=80,
            range=(0, 1500),
            label=dihiggs_01.title + '(mmc)',
            color='magenta',
            linestyle='solid',
            linewidth=2,
        histtype='step')
    if mmc_dihiggs_10 is not None:
        plt.hist(
            mmc_dihiggs_10[1],
            weights=dihiggs_10.ak_array['EventInfo___NominalAuxDyn']['evtweight'],
            bins=80,
            range=(0, 1500),
            label=dihiggs_10.title + '(mmc)',
            color='gray',
            linestyle='solid',
            linewidth=2,
        histtype='step')

    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/ztt_signal_distributions_overlay.pdf')
    plt.close(fig)


def signal_ztt_pt_overlay(ztautau, dihiggs_01, dihiggs_10):

    fig = plt.figure()
    plt.hist(
        ak.flatten(ztautau.ak_array['taus']['pt'] / 1000.),
        label=ztautau.title,
        bins = 30,
        range = (0, 300),
        linewidth = 2,
        histtype = 'step')
    plt.hist(
        ak.flatten(dihiggs_01.fold_1_array['taus']['pt'] / 1000.),
        label=dihiggs_01.title,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')
    plt.hist(
        ak.flatten(dihiggs_10.fold_1_array['taus']['pt'] / 1000.),
        label=dihiggs_10.title,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')
    plt.xlabel(r'Ztt taus $p_{T}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/ztt_signal_taus_pt_overlay.pdf')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(
        ak.flatten(ztautau.ak_array['bjets']['pt'] / 1000.),
        label=ztautau.title,
        bins = 30,
        range = (0, 300),
        linewidth = 2,
        histtype = 'step')
    plt.hist(
        ak.flatten(dihiggs_01.fold_1_array['bjets']['pt'] / 1000.),
        label=dihiggs_01.title,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')
    plt.hist(
        ak.flatten(dihiggs_10.fold_1_array['bjets']['pt'] / 1000.),
        label=dihiggs_10.title,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')
    plt.xlabel(r'Ztt b-jets $p_{T}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/ztt_signal_bjets_pt_overlay.pdf')
    plt.close(fig)
