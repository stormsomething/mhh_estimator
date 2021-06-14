import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
import scipy.stats as sc
import numpy as np
hep.set_style("ATLAS")

mpl.rc('font', **{'family':'serif','serif':['Palatino']})
mpl.rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
mpl.rc('text', usetex=True)    # mpl.rcParams['text.latex.unicode'] = True

from . import log; log = log.getChild(__name__)


def signal_pred_target_comparison(
        predictions_HH_10,
        predictions_HH_01,
        test_target_HH_10,
        test_target_HH_01,
        dihiggs_10,
        dihiggs_01,
        label,
        regressor='keras'):

    log.info('plotting distributions')

    fig = plt.figure()
    plt.hist(
        predictions_HH_10,
        bins=80,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_10.title + ' - pred.',
        color=dihiggs_10.color,
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        test_target_HH_10,
        bins=80,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_10.title + ' - truth.',
        color=dihiggs_10.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        predictions_HH_01,
        bins=80,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_01.title + ' - pred.',
        color=dihiggs_01.color,
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        test_target_HH_01,
        bins=80,
        # bins=15,
        range=(0, 1500), # (0.9, 1.1)
        label=dihiggs_01.title + ' - truth.',
        color=dihiggs_01.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3, title=regressor)
    fig.savefig('plots/' + str(label) + '_distributions_{}.pdf'.format(regressor))
    plt.close(fig)

    fig = plt.figure()
    plt.hist(
        test_target_HH_10,
        bins=80,
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
        range=(0, 1500),
        label=dihiggs_01.title,
        color=dihiggs_01.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')

    plt.xlabel(r'True $m_{hh}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_distributions_truthonly.pdf')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(
        [
            predictions_HH_01 - test_target_HH_01,
            predictions_HH_10 - test_target_HH_10,
        ],
        label=[
            dihiggs_01.title,
            dihiggs_10.title,
        ],
        color=[
            dihiggs_01.color,
            dihiggs_10.color,
        ],
        bins=160,
        range=(-400, 400),
        linewidth=2,
        histtype='step')
    plt.xlabel(r'$m_{hh}$: prediction - truth [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + '_deltas_{}.pdf'.format(regressor))
    plt.close(fig)

    fig = plt.figure()

    ratio_HH_01 = predictions_HH_01 / test_target_HH_01
    ratio_HH_10 = predictions_HH_10 / test_target_HH_10

    avg_ratio_HH_01 = sum(ratio_HH_01) / len(ratio_HH_01)
    avg_ratio_HH_10 = sum(ratio_HH_10) / len(ratio_HH_10)

    rms_ratio_HH_01 = np.sqrt(np.mean(ratio_HH_01**2))
    rms_ratio_HH_10 = np.sqrt(np.mean(ratio_HH_10**2))

    from sklearn.metrics import mean_squared_error

    pred_ratio_HH_01 = []
    for i in range(len(ratio_HH_01)):
        pred_ratio_HH_01.append(1.0)
        i = i + 1

    pred_ratio_HH_10 = []
    for i in range(len(ratio_HH_10)):
        pred_ratio_HH_10.append(1.0)
        i = i + 1

    from math import sqrt
    rmse_ratio_HH_01 = sqrt(mean_squared_error(ratio_HH_01, pred_ratio_HH_01))
    rmse_ratio_HH_10 = sqrt(mean_squared_error(ratio_HH_10, pred_ratio_HH_10))

    plt.hist(
        [
            ratio_HH_01,
            ratio_HH_10
        ],
        label=[
            r'$\kappa_{\lambda}$ = 1. Mean: ' + str(round(avg_ratio_HH_01, 4)) + '. RMS: ' + str(round(rms_ratio_HH_01, 4)) + '. RMSE: ' + str(round(rmse_ratio_HH_01, 4)) + '.', # dihiggs_01.title
            r'$\kappa_{\lambda}$ = 10. Mean: ' + str(round(avg_ratio_HH_10, 4)) + '. RMS: ' + str(round(rms_ratio_HH_10, 4)) + '. RMSE: ' + str(round(rmse_ratio_HH_10, 4)) + '.', # dihiggs_10.title
        ],
        color=[
            dihiggs_01.color,
            dihiggs_10.color,
        ],
        bins=160,
        range=(0., 3.),
        linewidth=2,
         histtype='step')
    plt.xlabel(r'$m_{hh}$: prediction / truth [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3, title=regressor)
    fig.savefig('plots/' + str(label) + '_ratios_{}.pdf'.format(regressor))
    plt.close(fig)


def signal_features(dihiggs_01, dihiggs_10):

    fig = plt.figure()
    plt.hist(
        ak.flatten(dihiggs_01.fold_1_array['taus']['pt'] / 1000.),
        label=dihiggs_01.title,
        color=dihiggs_01.color,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')
    plt.hist(
        ak.flatten(dihiggs_10.fold_1_array['taus']['pt'] / 1000.),
        label=dihiggs_10.title,
        color=dihiggs_10.color,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')
    plt.xlabel(r'Hadronic taus $p_{T}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/hadronic_taus_pt.pdf')
    plt.close(fig)

    fig = plt.figure()
    plt.hist(
        ak.flatten(dihiggs_01.fold_1_array['bjets']['pt'] / 1000.),
        label=dihiggs_01.title,
        color=dihiggs_01.color,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')
    plt.hist(
        ak.flatten(dihiggs_10.fold_1_array['bjets']['pt'] / 1000.),
        label=dihiggs_10.title,
        color=dihiggs_10.color,
        bins=30,
        range=(0, 300),
        linewidth=2,
        histtype='step')
    plt.xlabel(r'b-jets $p_{T}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/hadronic_bjets_pt.pdf')
    plt.close(fig)

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

def calculate_eff(n):

    tot = 0
    for N in range(len(n)):
        tot = tot + n[N]
        N = N + 1

    N = 79
    eff = []
    subtot = 0
    while N in range(len(n)):
        subtot = subtot + n[N]
        eff.append(subtot/tot)
        N = N - 1
    eff.reverse()

    return eff

def roc_plot(eff_01, eff_10):

    fig = plt.figure()
    plt.plot(eff_01[0], eff_10[0], color = 'blue', label = 'scikit')
    plt.plot(eff_01[1], eff_10[1], color = 'red', label = 'keras')
    plt.plot(eff_01[2], eff_10[2], color = 'green', label = 'truth')
    if len(eff_01) == 4:
        plt.plot(eff_01[3], eff_10[3], color = 'purple', label = 'mmc')
    plt.xlabel(r'$\kappa_{\lambda}$ = 1 Efficiency')
    plt.ylabel(r'$\kappa_{\lambda}$ = 10 Efficiency')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.legend()
    fig.savefig('plots/comparing_roc_curves.pdf')
    plt.close(fig)

def avg_rms_mhh_calculation(dihiggs, truth, scikit, keras, mmc):

    avg_mu = dihiggs.fold_1_array['EventInfo___NominalAuxDyn.CorrectedAndScaledAvgMu']

    avg_mhh_truth = []
    rms_mhh_truth = []
    mhh_truth_sem = []

    avg_mhh_scikit = []
    rms_mhh_scikit = []
    mhh_scikit_sem = []

    avg_mhh_keras = []
    rms_mhh_keras = []
    mhh_keras_sem = []

    if mmc is not None:
        avg_mhh_mmc = []
        rms_mhh_mmc = []
        mhh_mmc_sem = []

    bins = [[0,20], [20,30], [30,40], [40,50], [50,60], [60,85]]

    i = 0
    while i in range(len(bins)):
        mod_truth = []
        mod_scikit = []
        mod_keras = []
        if mmc is not None:
            mod_mmc = []
        for n in range(len(truth)):
            if avg_mu[n] >= bins[i][0] and avg_mu[n] < bins[i][1]:
                mod_truth.append(truth[n])
                mod_scikit.append(scikit[n])
                mod_keras.append(keras[n])
                if mmc is not None:
                    mod_mmc.append(mmc[n])
            n = n + 1

        avg_truth = sum(mod_truth) / len(mod_truth)
        sem_truth = sc.sem(mod_truth)
        avg_mhh_truth.append(avg_truth)
        mhh_truth_sem.append(sem_truth)
        mod_truth = np.array(mod_truth)
        rms_truth = np.sqrt(np.mean(mod_truth**2))
        rms_mhh_truth.append(rms_truth)

        avg_scikit = sum(mod_scikit) / len(mod_scikit)
        sem_scikit = sc.sem(mod_scikit)
        avg_mhh_scikit.append(avg_scikit)
        mhh_scikit_sem.append(sem_scikit)
        mod_scikit = np.array(mod_scikit)
        rms_scikit = np.sqrt(np.mean(mod_scikit**2))
        rms_mhh_scikit.append(rms_scikit)

        avg_keras = sum(mod_keras) / len(mod_keras)
        sem_keras = sc.sem(mod_keras)
        avg_mhh_keras.append(avg_keras)
        mhh_keras_sem.append(sem_keras)
        mod_keras = np.array(mod_keras)
        rms_keras = np.sqrt(np.mean(mod_keras**2))
        rms_mhh_keras.append(rms_keras)

        if mmc is not None:
            avg_mmc = sum(mod_mmc) / len(mod_mmc)
            sem_mmc = sc.sem(mod_mmc)
            avg_mhh_mmc.append(avg_mmc)
            mhh_mmc_sem.append(sem_mmc)
            mod_mmc = np.array(mod_mmc)
            rms_mmc = np.sqrt(np.mean(mod_mmc**2))
            rms_mhh_mmc.append(rms_mmc)

        i = i + 1

    avg_mhh = [avg_mhh_truth, mhh_truth_sem, avg_mhh_scikit, mhh_scikit_sem, avg_mhh_keras, mhh_keras_sem]
    if mmc is not None:
        avg_mhh.append(avg_mhh_mmc)
        avg_mhh.append(mhh_mmc_sem)

    rms_mhh = [rms_mhh_truth, rms_mhh_scikit, rms_mhh_keras]
    if mmc is not None:
        rms_mhh.append(rms_mhh_mmc)

    return [avg_mhh, rms_mhh]

def avg_mhh_plot(avg_mhh, name, sample, num):

    x = [0, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 85]
    y_truth = []
    y_scikit = []
    y_keras = []
    if len(avg_mhh) == 8:
        y_mmc = []
    for i in range(int(len(x)/2)):
        y_truth.append(avg_mhh[0][i])
        y_truth.append(avg_mhh[0][i])
        y_scikit.append(avg_mhh[2][i])
        y_scikit.append(avg_mhh[2][i])
        y_keras.append(avg_mhh[4][i])
        y_keras.append(avg_mhh[4][i])
        if len(avg_mhh) == 8:
            y_mmc.append(avg_mhh[6][i])
            y_mmc.append(avg_mhh[6][i])
        i = i + 1
    fig = plt.figure()
    plt.plot(x, y_scikit, color = 'blue', label = 'scikit')

    plt.plot(x, y_keras, color = 'red', label = 'keras')
    plt.plot(x, y_truth, color = 'green', label = 'truth')
    if len(avg_mhh) == 8:
        plt.plot(x, y_mmc, color = 'purple', label = 'mmc')
    if len(avg_mhh) == 6:
        if num == 1:
            plt.ylim(485, 545)
        if num == 10:
            plt.ylim(370, 410)
    if len(avg_mhh) == 8:
        if num == 1:
            plt.ylim(400, 540)
        if num == 10:
            plt.ylim(300, 410)
    plot_error_bars(avg_mhh)
    plt.xlim(0, 85)
    plt.xlabel('Average number of pp interactions per bunch crossing')
    plt.ylabel(r'$<m_{hh}>$ in each bin +/- SEM [GeV]')
    plt.legend(title = sample.title)
    fig.savefig('plots/' + name + '.pdf')
    plt.close(fig)

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
        i = i + 1
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

def plot_error_bars(avg_mhh):

    x = [10, 25, 35, 45, 55, 72.5]
    y_truth = avg_mhh[0]
    truth_err = avg_mhh[1]
    y_scikit = avg_mhh[2]
    scikit_err = avg_mhh[3]
    y_keras = avg_mhh[4]
    keras_err = avg_mhh[5]
    if len(avg_mhh) == 8:
        y_mmc = avg_mhh[6]
        mmc_err = avg_mhh[7]
    plt.errorbar(x, y_truth, yerr = truth_err, fmt = 'none', ecolor = 'green', capsize = 6)
    plt.errorbar(x, y_scikit, yerr = scikit_err, fmt = 'none', ecolor = 'blue', capsize = 6)
    plt.errorbar(x, y_keras, yerr = keras_err, fmt = 'none', ecolor = 'red', capsize = 6)
    if len(avg_mhh) == 8:
        plt.errorbar(x, y_mmc, yerr = mmc_err, fmt = 'none', ecolor = 'purple', capsize = 6)
