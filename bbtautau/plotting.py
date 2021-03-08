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
        dihiggs_01):

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
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/distributions.pdf')
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
        histtype='step')
    plt.xlabel(r'$m_{hh}$: prediction - truth [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/deltas.pdf')
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
        histtype='step')
    plt.xlabel(r'$m_{hh}$: prediction / truth [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/ratios.pdf')
    plt.close(fig)


def signal_features(dihiggs_01, dihiggs_10):

    fig = plt.figure()
    plt.hist(
        ak.flatten(dihiggs_01.fold_1_array['taus']['pt'] / 1000.),
        label=dihiggs_01.title,
        color=dihiggs_01.color,
        bins=30,
        range=(0, 300),
        histtype='step')
    plt.hist(
        ak.flatten(dihiggs_10.fold_1_array['taus']['pt'] / 1000.),
        label=dihiggs_10.title,
        color=dihiggs_10.color,
        bins=30,
        range=(0, 300),
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
        histtype='step')
    plt.hist(
        ak.flatten(dihiggs_10.fold_1_array['bjets']['pt'] / 1000.),
        label=dihiggs_10.title,
        color=dihiggs_10.color,
        bins=30,
        range=(0, 300),
        histtype='step')
    plt.xlabel(r'b-jets $p_{T}$ [GeV]')
    plt.ylabel('Raw Simulation Entries')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/hadronic_bjets_pt.pdf')
    plt.close(fig)
