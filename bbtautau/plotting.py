import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep as hep
import awkward as ak
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
        regressor='keras'):

    log.info('plotting distributions')
    fig = plt.figure()
    plt.hist(
        predictions_HH_10,
        bins=80,
        range=(0, 1500),
        label=dihiggs_10.title + ' - pred.',
        color=dihiggs_10.color,
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        test_target_HH_10,
        bins=80,
        range=(0, 1500),
        label=dihiggs_10.title + ' - truth.',
        color=dihiggs_10.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        predictions_HH_01,
        bins=80,
        range=(0, 1500),
        label=dihiggs_01.title + ' - pred.',
        color=dihiggs_01.color,
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    plt.hist(
        test_target_HH_01,
        bins=80,
        range=(0, 1500),
        label=dihiggs_01.title + ' - truth.',
        color=dihiggs_01.color,
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    
    plt.xlabel(r'$m_{hh}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3, title=regressor)
    fig.savefig('plots/distributions_{}.pdf'.format(regressor))
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
    fig.savefig('plots/distributions_truthonly.pdf')
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
    fig.savefig('plots/deltas_{}.pdf'.format(regressor))
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        [
            predictions_HH_01 / test_target_HH_01,
            predictions_HH_10 / test_target_HH_10
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
        range=(0., 3.),
        linewidth=2,
         histtype='step')
    plt.xlabel(r'$m_{hh}$: prediction / truth [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3, title=regressor)
    fig.savefig('plots/ratios_{}.pdf'.format(regressor))
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
    fig = plt.figure()
    plt.hist(
        scikit,
        bins=80,
        range=(0, 1500),
        label='scikit',
        color='red',
        linestyle='solid',
        linewidth=2,
        histtype='step')
    plt.hist(
        keras,
        bins=80,
        range=(0, 1500),
        label='keras',
        color='blue',
        linestyle='solid',
        linewidth=2,
        # cumulative=True,
        histtype='step')
    if type(mmc) != type(None):
        plt.hist(
            mmc,
            bins=80,
            range=(0, 1500),
            label='MMC',
            color='green',
            linestyle='solid',
            linewidth=2,
        histtype='step')

    plt.hist(
        truth,
        bins=80,
        range=(0, 1500),
        label='truth.',
        color='black',
        linestyle='dashed',
        linewidth=2,
        # cumulative=True,
        histtype='step')


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
