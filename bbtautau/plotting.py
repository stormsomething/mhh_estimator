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

def separation_overlay_plot(mhh_HH_01, mhh_HH_10, fold_1_array_1, fold_1_array_10, indices_1_1, indices_1_10, indices_2_1, indices_2_10):
    truths_1 = fold_1_array_1['universal_true_mhh']
    truths_10 = fold_1_array_10['universal_true_mhh']
    
    weights_1 = fold_1_array_1['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array_1['fold_weight']
    weights_10 = fold_1_array_10['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array_10['fold_weight']
    weights_10 = np.array(weights_10) / 11.8944157416 # Normalization factor calculated by hand
    
    mean_1 = np.mean(mhh_HH_01)
    mean_10 = np.mean(mhh_HH_10)
    rms_1 = np.sqrt(np.mean((mhh_HH_01 - mean_1) * (mhh_HH_01 - mean_1)))
    rms_10 = np.sqrt(np.mean((mhh_HH_10 - mean_10) * (mhh_HH_10 - mean_10)))
    
    fig = plt.figure()
    plt.hist(
        mhh_HH_01,
        bins=75,
        weights=weights_1,
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=1$')
    plt.hist(
        mhh_HH_10,
        bins=75,
        weights=weights_10,
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=10$')
    plt.hist(
        truths_1,
        bins=75,
        weights=weights_1,
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=1$')
    plt.hist(
        truths_10,
        bins=75,
        weights=weights_10,
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=10$')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='x-small')
    fig.savefig('plots/separation_overlay_truth_MDN.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        mhh_HH_01,
        bins=75,
        weights=weights_1,
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=1$')
    plt.hist(
        mhh_HH_10,
        bins=75,
        weights=weights_10,
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=10$')
    plt.hist(
        mhh_HH_01[indices_1_1],
        bins=75,
        weights=weights_1[indices_1_1],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=1$ (Relative $\sigma<$Resolution)')
    plt.hist(
        mhh_HH_10[indices_1_10],
        bins=75,
        weights=weights_10[indices_1_10],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=10$ (Relative $\sigma<$Resolution)')
    plt.hist(
        mhh_HH_01[indices_2_1],
        bins=75,
        weights=weights_1[indices_2_1],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=1$ (Relative $\sigma>$Resolution)')
    plt.hist(
        mhh_HH_10[indices_2_10],
        bins=75,
        weights=weights_10[indices_2_10],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=10$ (Relative $\sigma>$Resolution)')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='x-small')
    fig.savefig('plots/separation_overlay_sigma_category.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        mhh_HH_01[indices_1_1],
        bins=75,
        weights=weights_1[indices_1_1],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=1$ (Relative $\sigma<$Resolution)',
        color='red')
    plt.hist(
        mhh_HH_10[indices_1_10],
        bins=75,
        weights=weights_10[indices_1_10],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=10$ (Relative $\sigma<$Resolution)',
        color='blue')
    plt.hist(
        truths_1[indices_1_1],
        bins=75,
        weights=weights_1[indices_1_1],
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=1$ (Relative $\sigma<$Resolution)',
        color='red',
        linestyle='dashed')
    plt.hist(
        truths_10[indices_1_10],
        bins=75,
        weights=weights_10[indices_1_10],
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=10$ (Relative $\sigma<$Resolution)',
        color='blue',
        linestyle='dashed')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='x-small')
    fig.savefig('plots/separation_overlay_low_sigma.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        mhh_HH_01[indices_2_1],
        bins=75,
        weights=weights_1[indices_2_1],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=1$ (Relative $\sigma>$Resolution)',
        color='red')
    plt.hist(
        mhh_HH_10[indices_2_10],
        bins=75,
        weights=weights_10[indices_2_10],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=10$ (Relative $\sigma>$Resolution)',
        color='blue')
    plt.hist(
        truths_1[indices_2_1],
        bins=75,
        weights=weights_1[indices_2_1],
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=1$ (Relative $\sigma>$Resolution)',
        color='red',
        linestyle='dashed')
    plt.hist(
        truths_10[indices_2_10],
        bins=75,
        weights=weights_10[indices_2_10],
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=10$ (Relative $\sigma>$Resolution)',
        color='blue',
        linestyle='dashed')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='x-small')
    fig.savefig('plots/separation_overlay_high_sigma.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        mhh_HH_01[indices_1_1],
        bins=75,
        weights=weights_1[indices_1_1],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=1$ (Relative $\sigma<$Resolution)',
        color='red')
    plt.hist(
        mhh_HH_10[indices_1_10],
        bins=75,
        weights=weights_10[indices_1_10],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=10$ (Relative $\sigma<$Resolution)',
        color='blue')
    plt.hist(
        truths_1[indices_1_1],
        bins=75,
        weights=weights_1[indices_1_1],
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=1$ (Relative $\sigma<$Resolution)',
        color='red',
        linestyle='dashed')
    plt.hist(
        truths_10[indices_1_10],
        bins=75,
        weights=weights_10[indices_1_10],
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=10$ (Relative $\sigma<$Resolution)',
        color='blue',
        linestyle='dashed')
    plt.hist(
        mhh_HH_01[indices_2_1],
        bins=75,
        weights=weights_1[indices_2_1],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=1$ (Relative $\sigma>$Resolution)',
        color='orange')
    plt.hist(
        mhh_HH_10[indices_2_10],
        bins=75,
        weights=weights_10[indices_2_10],
        range=(0,1500),
        histtype='step',
        label=r'MDN $\kappa_\lambda=10$ (Relative $\sigma>$Resolution)',
        color='green')
    plt.hist(
        truths_1[indices_2_1],
        bins=75,
        weights=weights_1[indices_2_1],
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=1$ (Relative $\sigma>$Resolution)',
        color='orange',
        linestyle='dashed')
    plt.hist(
        truths_10[indices_2_10],
        bins=75,
        weights=weights_10[indices_2_10],
        range=(0,1500),
        histtype='step',
        label=r'Truth $\kappa_\lambda=10$ (Relative $\sigma>$Resolution)',
        color='green',
        linestyle='dashed')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='x-small')
    fig.savefig('plots/separation_overlay_both_sigma.pdf')
    plt.close(fig)

def simple_sigma_plot(sigmas, fold_1_array, label):
    weights = fold_1_array['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array['fold_weight']
    
    trimmed_sigmas = sigmas[np.where(sigmas < 100)]
    
    mean_sigma = np.mean(trimmed_sigmas)
    rms_sigma = np.sqrt(np.mean((trimmed_sigmas - mean_sigma) * (trimmed_sigmas - mean_sigma)))
    median_sigma = np.median(sigmas)
    
    # Split indices on sigma ranges
    indices_1 = np.where(sigmas < 1)
    indices_2 = np.where(sigmas > 1)
    
    fig = plt.figure()
    plt.hist(
        sigmas,
        bins=30,
        weights=weights,
        range=(0,6),
        label='Mean: ' + str(round(mean_sigma, 4)) + '. Median: ' + str(round(median_sigma, 4)) + '. RMS: ' + str(round(rms_sigma, 4)))
    plt.hist(
        sigmas[indices_1],
        bins=30,
        weights=weights[indices_1],
        range=(0,6))
    plt.hist(
        sigmas[indices_2],
        bins=30,
        weights=weights[indices_2],
        range=(0,6))
    plt.xlim((0,6))
    plt.ylim(bottom=0)
    plt.xlabel(r'Relative $\sigma$ / Resolution')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/rel_sigma_over_resol_' + label + '.pdf')
    plt.close(fig)

def res_plots(mus, sigmas, fold_1_array, label):
    weights = fold_1_array['EventInfo___NominalAuxDyn']['evtweight']*fold_1_array['fold_weight']
    res = mus / fold_1_array['universal_true_mhh']
    rel_sigmas = sigmas / mus
    
    indices_low = np.where(fold_1_array['universal_true_mhh'] < 400)
    indices_mid = np.where((fold_1_array['universal_true_mhh'] > 400) & (fold_1_array['universal_true_mhh'] < 800))
    indices_high = np.where(fold_1_array['universal_true_mhh'] > 800)
    
    fig = plt.figure()
    plt.hist(
        res,
        bins=40,
        weights=weights,
        range=(0,2),
        label='Mean: ' + str(round(np.mean(res), 4)))
    plt.xlim((0,2))
    plt.ylim(bottom=0)
    plt.xlabel(r'MDN $m_{HH}$/Truth $m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/response_dist_all_mhh_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        res[indices_low],
        bins=40,
        weights=weights[indices_low],
        range=(0,2),
        label='Mean: ' + str(round(np.mean(res[indices_low]), 4)))
    plt.xlim((0,2))
    plt.ylim(bottom=0)
    plt.xlabel(r'MDN $m_{HH}$/Truth $m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/response_dist_low_mhh_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        res[indices_mid],
        bins=40,
        weights=weights[indices_mid],
        range=(0,2),
        label='Mean: ' + str(round(np.mean(res[indices_mid]), 4)))
    plt.xlim((0,2))
    plt.ylim(bottom=0)
    plt.xlabel(r'MDN $m_{HH}$/Truth $m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/response_dist_mid_mhh_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        res[indices_high],
        bins=40,
        weights=weights[indices_high],
        range=(0,2),
        label='Mean: ' + str(round(np.mean(res[indices_high]), 4)))
    plt.xlim((0,2))
    plt.ylim(bottom=0)
    plt.xlabel(r'MDN $m_{HH}$/Truth $m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/response_dist_high_mhh_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        rel_sigmas,
        bins=40,
        weights=weights,
        range=(0,0.8),
        label='Mean: ' + str(round(np.mean(rel_sigmas), 4)))
    plt.xlim((0,0.8))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\sigma (m_{HH})/m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/sigma_dist_all_mhh_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        rel_sigmas[indices_low],
        bins=40,
        weights=weights[indices_low],
        range=(0,0.8),
        label='Mean: ' + str(round(np.mean(rel_sigmas[indices_low]), 4)))
    plt.xlim((0,0.8))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\sigma (m_{HH})/m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/sigma_dist_low_mhh_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        rel_sigmas[indices_mid],
        bins=40,
        weights=weights[indices_mid],
        range=(0,0.8),
        label='Mean: ' + str(round(np.mean(rel_sigmas[indices_mid]), 4)))
    plt.xlim((0,0.8))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\sigma (m_{HH})/m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/sigma_dist_mid_mhh_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        rel_sigmas[indices_high],
        bins=40,
        weights=weights[indices_high],
        range=(0,0.8),
        label='Mean: ' + str(round(np.mean(rel_sigmas[indices_high]), 4)))
    plt.xlim((0,0.8))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\sigma (m_{HH})/m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/sigma_dist_high_mhh_' + label + '.pdf')
    plt.close(fig)

def eta_plot(higgs_array, weights, label):
    """
    print(higgs_array)
    print(higgs_array['px'])
    print(np.array(higgs_array['px']).shape)
    """
    px_tot = np.array(np.sum(higgs_array['px'], axis=1))
    py_tot = np.array(np.sum(higgs_array['py'], axis=1))
    pz_tot = np.array(np.sum(higgs_array['pz'], axis=1))
    p_tot = np.sqrt(px_tot * px_tot + py_tot * py_tot + pz_tot * pz_tot)
    eta = np.arctanh(pz_tot / p_tot)
    
    trimmed_eta = eta[np.where(np.abs(eta) < 100)]
    mean_eta = np.mean(trimmed_eta)
    rms_eta = np.sqrt(np.mean((trimmed_eta - mean_eta) * (trimmed_eta - mean_eta)))
    median_eta = np.median(eta)
    
    """
    print(px_tot)
    print(py_tot)
    print(px_tot.shape)
    print(py_tot.shape)
    print(pz_tot.shape)
    print(p_tot.shape)
    print(eta.shape)
    print(weights.shape)
    """

    fig = plt.figure()
    plt.hist(
        eta,
        bins=64,
        weights=weights,
        range=(-8,8),
        label='Mean: ' + str(round(mean_eta, 4)) + '. Median: ' + str(round(median_eta, 4)) + '. RMS: ' + str(round(rms_eta, 4)))
    plt.xlim((-8,8))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\eta_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/eta_' + label + '.pdf')
    plt.close(fig)
    
    mean_eta_abs = np.mean(np.abs(trimmed_eta))
    rms_eta_abs = np.sqrt(np.mean((np.abs(trimmed_eta) - mean_eta_abs) * (np.abs(trimmed_eta) - mean_eta_abs)))
    median_eta_abs = np.median(np.abs(trimmed_eta))

    fig = plt.figure()
    plt.hist(
        np.abs(eta),
        bins=32,
        weights=weights,
        range=(0,8),
        label='Mean: ' + str(round(mean_eta_abs, 4)) + '. Median: ' + str(round(median_eta_abs, 4)) + '. RMS: ' + str(round(rms_eta_abs, 4)))
    plt.xlim((0,8))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\eta_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/eta_abs_' + label + '.pdf')
    plt.close(fig)
    
    pt = np.sqrt(px_tot * px_tot + py_tot * py_tot)
    
    #trimmed_pt = pt[np.where(pt < 1000000)]
    mean_pt = np.mean(pt)
    rms_pt = np.sqrt(np.mean((pt - mean_pt) * (pt - mean_pt)))
    median_pt = np.median(pt)
    
    fig = plt.figure()
    plt.hist(
        pt,
        bins=60,
        weights=weights,
        range=(0,300000),
        label='Mean: ' + str(round(mean_pt, 4)) + '. Median: ' + str(round(median_pt, 4)) + '. RMS: ' + str(round(rms_pt, 4)))
    plt.xlim((0,300000))
    plt.ylim(bottom=0)
    plt.xlabel(r'$p_{T,HH}$ (MeV)')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/pTHH_' + label + '.pdf')
    plt.close(fig)

def klambda_scan_plot(klambdas, truth_significances, mdn_significances, mmc_significances, split_significances, k10mode = False, bonus_pts = None, split_truth = None):
    fig = plt.figure()
    plt.plot(
        klambdas,
        truth_significances,
        label='Truth',
        color='black')
    plt.plot(
        klambdas,
        mdn_significances,
        label='MDN',
        color='red')
    plt.plot(
        klambdas,
        mmc_significances,
        label='MMC',
        color='purple')
    plt.plot(
        klambdas,
        split_significances,
        label=r'MDN Split by Relative $\sigma$/Resolution',
        color='green')
    if (split_truth is not None):
        plt.plot(
            klambdas,
            split_truth,
            label=r'Truth Split by Relative $\sigma$/Resolution',
            color='pink')
    if (bonus_pts is not None):
        if (k10mode):
            plt.scatter(
                1,
                bonus_pts[0],
                label='Truth',
                color='black',
                marker='o')
            plt.scatter(
                1,
                bonus_pts[1],
                label='MDN',
                color='red',
                marker='o')
            plt.scatter(
                1,
                bonus_pts[2],
                label='MMC',
                color='purple',
                marker='o')
            plt.scatter(
                1,
                bonus_pts[3],
                label=r'MDN Split by Relative $\sigma$/Resolution',
                color='green',
                marker='o')
            if (split_truth is not None):
                plt.scatter(
                    1,
                    bonus_pts[4],
                    label=r'Truth Split by Relative $\sigma$/Resolution',
                    color='pink',
                    marker='o')
        else:
            plt.scatter(
                10,
                bonus_pts[0],
                label='Truth',
                color='black',
                marker='o')
            plt.scatter(
                10,
                bonus_pts[1],
                label='MDN',
                color='red',
                marker='o')
            plt.scatter(
                10,
                bonus_pts[2],
                label='MMC',
                color='purple',
                marker='o')
            plt.scatter(
                10,
                bonus_pts[3],
                label=r'MDN Split by Relative $\sigma$/Resolution',
                color='green',
                marker='o')
            if (split_truth is not None):
                plt.scatter(
                    10,
                    bonus_pts[4],
                    label=r'Truth Split by Relative $\sigma$/Resolution',
                    color='pink',
                    marker='o')
    plt.xlim((klambdas[0],klambdas[-1]))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\kappa_\lambda$')
    plt.legend(fontsize='small')
    if (k10mode):
        plt.ylabel(r'Separation Significance Compared to $\kappa_\lambda=10$')
        fig.savefig('plots/klambda_scan_from_k10.pdf')
    else:
        plt.ylabel(r'Separation Significance Compared to $\kappa_\lambda=1$')
        fig.savefig('plots/klambda_scan.pdf')
    plt.close(fig)

def reweight_and_compare(mhh, original_weights, new_weights, label, klambda, cs_norm = None, k10mode = False):
    # Option to renormalize so that both signals have the same total weights
    if (cs_norm is None):
        cs_norm = sum(original_weights) / sum(new_weights)
    new_weights = np.array(new_weights) * cs_norm
    
    fig = plt.figure()
    if (k10mode):
        (n_01, bins_01, patches_01) = plt.hist(
            mhh,
            bins=40,
            weights=original_weights,
            range=(200,1000),
            histtype='step',
            label=r'$\kappa_\lambda=10$')
    else:
        (n_01, bins_01, patches_01) = plt.hist(
            mhh,
            bins=40,
            weights=original_weights,
            range=(200,1000),
            histtype='step',
            label=r'$\kappa_\lambda=1$')
    (n_10, bins_10, patches_10) = plt.hist(
        mhh,
        bins=40,
        weights=new_weights,
        range=(200,1000),
        histtype='step',
        label=r'$\kappa_\lambda=$' + klambda)
        
    sig_sum = 0
    for i in range(len(n_01)):
        if (n_01[i] > 0) and (n_10[i] > 0):
            sig_sum += (n_10[i] * np.log(n_10[i] / n_01[i]) - n_10[i] + n_01[i])
    z = np.sqrt(2 * sig_sum)
    
    if (k10mode):
        plt.hist(
            [0],
            bins=1,
            weights=[n_01[1]],
            range=(0,1500),
            label=r'$\kappa_\lambda=$' + klambda + ' Hypothesis Significance Compared to $\kappa_\lambda=10$: $Z = $' + str(round(z, 4)),
            color='white')
    else:
        plt.hist(
            [0],
            bins=1,
            weights=[n_01[1]],
            range=(0,1500),
            label=r'$\kappa_\lambda=$' + klambda + ' Hypothesis Significance Compared to $\kappa_\lambda=1$: $Z = $' + str(round(z, 4)),
            color='white')
    plt.hist(
        [0],
        bins=1,
        weights=[n_01[1]],
        range=(0,1500),
        label='Integral Scale Factor Between Signals: ' + str(round(cs_norm, 4)),
        color='white')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='x-small')
    """
    if (k10mode):
        fig.savefig('plots/reweight_and_compare_' + label + '_' + klambda + '_from_k10.pdf')
    else:
        fig.savefig('plots/reweight_and_compare_' + label + '_' + klambda + '.pdf')
    """
    plt.close(fig)
    
    return z, cs_norm

def resolution_plot(mus, sigmas, fold_1_array, label):
    weights = fold_1_array['EventInfo___NominalAuxDyn']['evtweight']*fold_1_array['fold_weight']

    res = mus / fold_1_array['universal_true_mhh']
    bins = np.arange(0,1500+1500./80.,1500./80.)
    # Use this to bin by regressed mHH
    bin_centers, bin_errors, means, mean_stat_err, resol = response_curve(res, mus, bins)
    _, _, _, _, resol_2 = response_curve(res, mus, bins, mode=1)
    _, _, _, _, resol_std = response_curve(res, mus, bins, mode=2)
    
    event_resol = []
    tot_sigma_by_bin = np.zeros(len(bins))
    tot_sigma_sq_by_bin = np.zeros(len(bins))
    tot_res_by_bin = np.zeros(len(bins))
    count_by_bin = np.zeros(len(bins))
    for i in range(len(mus)):
        bin_num = (np.floor(80 * mus[i] / 1500)).astype(int)
        if ((bin_num < 0) or (bin_num > len(bins) - 2)):
            event_resol += [0]
        else:
            if (bin_num < 16):
                event_resol += [resol[16]]
            else:
                event_resol += [resol[bin_num]]
            tot_sigma_by_bin[bin_num] += sigmas[i] / mus[i]
            tot_sigma_sq_by_bin[bin_num] += (sigmas[i] / mus[i]) ** 2
            tot_res_by_bin[bin_num] += mus[i] / fold_1_array['universal_true_mhh'][i]
            count_by_bin[bin_num] += 1
    
    mean_sigma_by_bin = np.zeros(len(bins) - 1)
    rms_sigma_by_bin = np.zeros(len(bins) - 1)
    mean_res_by_bin = np.zeros(len(bins) - 1)
    for j in range(len(bins) - 1):
        if (count_by_bin[j] == 0):
            mean_sigma_by_bin[j] = 0
            rms_sigma_by_bin[j] = 0
            mean_res_by_bin[j] = 0
        else:
            mean_sigma_by_bin[j] = tot_sigma_by_bin[j] / count_by_bin[j]
            rms_sigma_by_bin[j] = np.sqrt(tot_sigma_sq_by_bin[j] / count_by_bin[j])
            mean_res_by_bin[j] = tot_res_by_bin[j] / count_by_bin[j]
    
    mean_resol = np.mean(event_resol)
    rms_resol = np.sqrt(np.mean((event_resol - mean_resol) * (event_resol - mean_resol)))
    
    fig = plt.figure()
    plt.hist(
        event_resol,
        bins=30,
        weights=weights,
        range=(0,0.3),
        label='Mean: ' + str(round(mean_resol, 4)) + '. RMS: ' + str(round(rms_resol, 4)))
    plt.xlim((0,0.3))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ Resolution (GeV)')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/resolutions_' + label + '.pdf')
    plt.close(fig)
    
    sigma_over_resol = sigmas / (mus * event_resol)
    trimmed_sigma_over_resol = sigma_over_resol[np.where(np.array(event_resol) > 0)]
    mean_sigma_over_resol = np.mean(trimmed_sigma_over_resol)
    rms_sigma_over_resol = np.sqrt(np.mean((trimmed_sigma_over_resol - mean_sigma_over_resol) * (trimmed_sigma_over_resol - mean_sigma_over_resol)))
    
    """
    fig = plt.figure()
    plt.hist(
        sigma_over_resol,
        bins=30,
        weights=weights,
        range=(0,6),
        label='Mean: ' + str(round(mean_sigma_over_resol, 4)) + '. RMS: ' + str(round(rms_sigma_over_resol, 4)))
    plt.xlim((0,6))
    plt.ylim(bottom=0)
    plt.xlabel(r'Relative $\sigma$/Resolution')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/sigma_over_resol_' + label + '.pdf')
    plt.close(fig)
    """
    
    fig = plt.figure()
    plt.plot(
        bin_centers,
        resol,
        label='Resolution')
    plt.plot(
        bin_centers,
        mean_stat_err,
        label='Mean Statistical Error')
    plt.plot(
        bin_centers,
        mean_sigma_by_bin,
        label=r'Mean $\sigma(m_{HH})/m_{HH}$')
    plt.plot(
        bin_centers,
        rms_sigma_by_bin,
        label=r'Quad. Mean $\sigma(m_{HH})/m_{HH}$')
    plt.plot(
        bin_centers,
        resol_2,
        label=r'Resolution ($2\sigma$ Width Divided by 2)')
    plt.plot(
        bin_centers,
        resol_std,
        label='Resolution (Std. Dev.)')
    plt.xlim((0,1500))
    plt.ylim((0,0.7))
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.legend(fontsize='small')
    fig.savefig('plots/resolutions_by_bin_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(
        bin_centers,
        mean_res_by_bin,
        label=r'Mean Response')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.legend(fontsize='small')
    fig.savefig('plots/mean_response_by_bin_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(
        bin_centers,
        mean_sigma_by_bin / resol,
        label=r'Mean Relative $\sigma(m_{HH})$ / Resolution')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.legend(fontsize='small')
    fig.savefig('plots/sigma_over_resol_by_bin_' + label + '.pdf')
    plt.close(fig)
    
    # Use this to bin by true mHH
    bin_centers, bin_errors, means, mean_stat_err, resol = response_curve(res, fold_1_array['universal_true_mhh'], bins)
    _, _, _, _, resol_2 = response_curve(res, fold_1_array['universal_true_mhh'], bins, mode=1)
    _, _, _, _, resol_std = response_curve(res, fold_1_array['universal_true_mhh'], bins, mode=2)
    
    tot_sigma_by_bin = np.zeros(len(bins))
    tot_sigma_sq_by_bin = np.zeros(len(bins))
    tot_res_by_bin = np.zeros(len(bins))
    count_by_bin = np.zeros(len(bins))
    for i in range(len(mus)):
        bin_num = (np.floor(80 * fold_1_array['universal_true_mhh'][i] / 1500)).astype(int)
        if ((bin_num >= 0) and (bin_num <= len(bins) - 2)):
            tot_sigma_by_bin[bin_num] += sigmas[i] / mus[i]
            tot_sigma_sq_by_bin[bin_num] += (sigmas[i] / mus[i]) ** 2
            tot_res_by_bin[bin_num] += mus[i] / fold_1_array['universal_true_mhh'][i]
            count_by_bin[bin_num] += 1
    
    mean_sigma_by_bin = np.zeros(len(bins) - 1)
    rms_sigma_by_bin = np.zeros(len(bins) - 1)
    mean_res_by_bin = np.zeros(len(bins) - 1)
    for j in range(len(bins) - 1):
        if (count_by_bin[j] == 0):
            mean_sigma_by_bin[j] = 0
            rms_sigma_by_bin[j] = 0
            mean_res_by_bin[j] = 0
        else:
            mean_sigma_by_bin[j] = tot_sigma_by_bin[j] / count_by_bin[j]
            rms_sigma_by_bin[j] = np.sqrt(tot_sigma_sq_by_bin[j] / count_by_bin[j])
            mean_res_by_bin[j] = tot_res_by_bin[j] / count_by_bin[j]
            
    fig = plt.figure()
    plt.plot(
        bin_centers,
        resol,
        label='Resolution')
    plt.plot(
        bin_centers,
        mean_stat_err,
        label='Mean Statistical Error')
    plt.plot(
        bin_centers,
        mean_sigma_by_bin,
        label=r'Mean $\sigma(m_{HH})/m_{HH}$')
    plt.plot(
        bin_centers,
        rms_sigma_by_bin,
        label=r'Quad. Mean $\sigma(m_{HH})/m_{HH}$')
    plt.plot(
        bin_centers,
        resol_2,
        label=r'Resolution ($2\sigma$ Width Divided by 2)')
    plt.plot(
        bin_centers,
        resol_std,
        label='Resolution (Std. Dev.)')
    plt.xlim((0,1500))
    plt.ylim((0,0.7))
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.legend(fontsize='small')
    fig.savefig('plots/resolutions_by_truth_bin_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(
        bin_centers,
        mean_res_by_bin,
        label=r'Mean Response')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.legend(fontsize='small')
    fig.savefig('plots/mean_response_by_truth_bin_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.plot(
        bin_centers,
        mean_sigma_by_bin / resol,
        label=r'Mean Relative $\sigma(m_{HH})$ / Resolution')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.legend(fontsize='small')
    fig.savefig('plots/sigma_over_resol_by_truth_bin_' + label + '.pdf')
    plt.close(fig)

    return np.array(event_resol)

def get_quantile_width(arr, cl=0.68):
    q1 = (1. - cl) / 2.
    q2 = 1. - q1
    y = np.quantile(arr, [q1, q2])
    width = (y[1] - y[0]) / 2.
    return width

def response_curve(res, var, bins, mode=0):
    _bin_centers = []
    _bin_errors = []
    _means = []
    _mean_stat_err = []
    _resol = []
    for i in range(len(bins)-1):
        a = res[(var > bins[i]) & (var < bins[i+1])]
        if (len(a) == 0):
            _means += [0]
            _mean_stat_err += [0]
            _resol += [0]
        else:
            _means += [np.mean(a)]
            _mean_stat_err += [np.std(a, ddof=1) / np.sqrt(np.size(a))]
            if (mode == 1):
                _resol += [get_quantile_width(a, cl=0.95) / 2]
            elif (mode == 2):
                _resol += [np.std(a)]
            else:
                _resol += [get_quantile_width(a)]
        _bin_centers += [bins[i] + (bins[i+1] - bins[i]) / 2]
        _bin_errors += [(bins[i+1] - bins[i]) / 2]
    return np.array(_bin_centers), np.array(_bin_errors), np.array(_means), np.array(_mean_stat_err), np.array(_resol)
    
def reweight_plot(mhh_HH_01, mhh_HH_10, fold_1_array_1, fold_1_array_10, new_weights, label, slice_indices_1 = None, slice_indices_10 = None):
    weights_1 = fold_1_array_1['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array_1['fold_weight']
    weights_10 = fold_1_array_10['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array_10['fold_weight']
    
    if (slice_indices_1 is not None):
        weights_1 = weights_1[slice_indices_1]
        mhh_HH_01 = mhh_HH_01[slice_indices_1]
    if (slice_indices_10 is not None):
        weights_10 = weights_10[slice_indices_10]
        mhh_HH_10 = mhh_HH_10[slice_indices_10]
    
    fig, ax = plt.subplots()
    """
    (n_1, bins_1, patches_1) = plt.hist(
        mhh_HH_01,
        bins=40,
        weights=weights_1,
        range=(200,1000),
        histtype='step',
        label=r'$\kappa_\lambda=1$')
    """
    (n_10, bins_10, patches_10) = plt.hist(
        mhh_HH_10,
        bins=40,
        weights=weights_10,
        range=(200,1000),
        histtype='step',
        label=r'$\kappa_\lambda=10$')
    
    """
    counts_1 = n_1 * reweight_1[0] * norm
    bin_edges_1 = reweight_1[1]
    ax.bar(x=bin_edges_1[:-1], height=counts_1, width=np.diff(bin_edges_1), align='edge', label=r'$\kappa_\lambda=10$ (reweighted from $\kappa_\lambda=1$)', fill=False, edgecolor='green')
    counts_10 = n_10 * reweight_10[0] / norm
    bin_edges_10 = reweight_10[1]
    ax.bar(x=bin_edges_10[:-1], height=counts_10, width=np.diff(bin_edges_10), align='edge', label=r'$\kappa_\lambda=1$ (reweighted from $\kappa_\lambda=10$)', fill=False, edgecolor='red')
    """
    
    (counts_1, bins_1, patches_1) = plt.hist(
        mhh_HH_01,
        bins=40,
        weights=new_weights,
        range=(200,1000),
        histtype='step',
        label=r'$\kappa_\lambda=10$ (reweighted from $\kappa_\lambda=1$)')
    
    """
    print("My Histogram's Bins Are:")
    print(bins_1)
    print(bins_10)
    print("The Reweighting File Bins Are:")
    print(bin_edges_1)
    print(bin_edges_10)
    """
    
    plt.xlim((200,1000))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/k_lambda_reweight_' + label + '.pdf')
    plt.close(fig)
    
    ratios_1 = []
    ratios_10 = []
    for i in range(len(counts_1)):
        if (n_10[i] == 0):
            ratios_10.append(0)
        else:
            ratios_10.append(counts_1[i] / n_10[i])
        """
        if (n_1[i] <= 0):
            ratios_1.append(0)
        else:
            ratios_1.append(counts_10[i] / n_1[i])
        """
    
    fig, ax = plt.subplots()
    ax.bar(x=bins_10[:-1], height=ratios_10, width=np.diff(bins_10), align='edge', label=r'$\kappa_\lambda=10$')
    """
    counts_10 = n_10 * reweight_10[0] / norm
    bin_edges_10 = reweight_10[1]
    ax.bar(x=bin_edges_10[:-1], height=ratios_1, width=np.diff(bin_edges_10), align='edge', label=r'$\kappa_\lambda=1$', fill=False, edgecolor='red')
    """
    plt.xlim((200,1000))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Reweight/Truth Ratio')
    plt.legend(fontsize='small')
    fig.savefig('plots/k_lambda_reweight_ratio_' + label + '.pdf')
    plt.close(fig)
    
    
def k_lambda_comparison_plot(mhh_HH_01, mhh_HH_10, fold_1_array_1, fold_1_array_10, label, slice_indices_1 = None, slice_indices_10 = None, k10mode = False):
    weights_1 = fold_1_array_1['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array_1['fold_weight']
    weights_10 = fold_1_array_10['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array_10['fold_weight']
    
    print(label + " kappa_lambda=1 Weight Sum: " + str(sum(weights_1)))
    print(label + " kappa_lambda=10 Weight Sum: " + str(sum(weights_10)))
    if (k10mode):
        weights_10 = np.array(weights_10) * 11.8944157416 # Normalization factor calculated by hand
    else:
        weights_10 = np.array(weights_10) / 11.8944157416 # Normalization factor calculated by hand
    
    if (slice_indices_1 is not None):
        weights_1 = weights_1[slice_indices_1]
        mhh_HH_01 = mhh_HH_01[slice_indices_1]
    if (slice_indices_10 is not None):
        weights_10 = weights_10[slice_indices_10]
        mhh_HH_10 = mhh_HH_10[slice_indices_10]
    
    mean_1 = np.mean(mhh_HH_01)
    mean_10 = np.mean(mhh_HH_10)
    rms_1 = np.sqrt(np.mean((mhh_HH_01 - mean_1) * (mhh_HH_01 - mean_1)))
    rms_10 = np.sqrt(np.mean((mhh_HH_10 - mean_10) * (mhh_HH_10 - mean_10)))
    
    fig = plt.figure()
    (n_01, bins_01, patches_01) = plt.hist(
        mhh_HH_01,
        bins=75,
        weights=weights_1,
        range=(0,1500),
        histtype='step',
        label=r'$\kappa_\lambda=1$. Mean: ' + str(round(mean_1, 4)) + '. RMS: ' + str(round(rms_1, 4)))
    (n_10, bins_10, patches_10) = plt.hist(
        mhh_HH_10,
        bins=75,
        weights=weights_10,
        range=(0,1500),
        histtype='step',
        label=r'$\kappa_\lambda=10$. Mean: ' + str(round(mean_10, 4)) + '. RMS: ' + str(round(rms_10, 4)))
    sig_sum = 0
    for i in range(len(n_01)):
        #print(str(n_01[i]) + " ~~~ " + str(n_10[i]))
        if (n_01[i] > 0) and (n_10[i] > 0):
            sig_sum += (n_10[i] * np.log(n_10[i] / n_01[i]) - n_10[i] + n_01[i])
    z = np.sqrt(2 * sig_sum)
    plt.hist(
        [0],
        bins=1,
        weights=[n_01[1]],
        range=(0,1500),
        label=r'$\kappa_\lambda=10$ Hypothesis Significance Compared to $\kappa_\lambda=1$: $Z = $' + str(round(z, 4)),
        color='white')
    if (k10mode):
        plt.hist(
            [0],
            bins=1,
            weights=[n_01[1]],
            range=(0,1500),
            label='Integral Scale Factor Between Signals: ' + str(round(11.894416, 4)),
            color='white')
    else:
        plt.hist(
            [0],
            bins=1,
            weights=[n_01[1]],
            range=(0,1500),
            label='Integral Scale Factor Between Signals: ' + str(round(1 / 11.894416, 4)),
            color='white')
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='x-small')
    fig.savefig('plots/k_lambda_comparison_' + label + '.pdf')
    plt.close(fig)
    
    return z
    
def resid_comparison_plots(mus, sigmas, old_preds, mmc_preds, fold_1_array, label, mvis):
    truths = fold_1_array['universal_true_mhh']
    weights = fold_1_array['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array['fold_weight']

    # Split indices on absolute sigma ranges
    indices_1 = np.where(sigmas < 4)
    indices_2 = np.where(sigmas > 4)
    
    data = truths - mus
    mean_all = np.mean(data)
    rms_all = np.sqrt(np.mean((data - mean_all) * (data - mean_all)))
    
    old_data = truths - old_preds
    old_mean_1 = np.mean(old_data[indices_1])
    old_mean_2 = np.mean(old_data[indices_2])
    old_mean_all = np.mean(old_data)
    old_rms_1 = np.sqrt(np.mean((old_data[indices_1] - old_mean_1) * (old_data[indices_1] - old_mean_1)))
    old_rms_2 = np.sqrt(np.mean((old_data[indices_2] - old_mean_2) * (old_data[indices_2] - old_mean_2)))
    old_rms_all = np.sqrt(np.mean((old_data - old_mean_all) * (old_data - old_mean_all)))

    mmc_data = truths - mmc_preds
    mmc_mean_1 = np.mean(mmc_data[indices_1])
    mmc_mean_2 = np.mean(mmc_data[indices_2])
    mmc_mean_all = np.mean(mmc_data)
    mmc_rms_1 = np.sqrt(np.mean((mmc_data[indices_1] - mmc_mean_1) * (mmc_data[indices_1] - mmc_mean_1)))
    mmc_rms_2 = np.sqrt(np.mean((mmc_data[indices_2] - mmc_mean_2) * (mmc_data[indices_2] - mmc_mean_2)))
    mmc_rms_all = np.sqrt(np.mean((mmc_data - mmc_mean_all) * (mmc_data - mmc_mean_all)))
    
    fig = plt.figure()
    plt.hist(
        data,
        bins=60,
        weights=weights,
        range=(-500,500),
        histtype='step',
        label='All events, MDN. Mean: ' + str(round(mean_all, 4)) + '. RMS: ' + str(round(rms_all, 4)),
        density=True,
        color='red')
    plt.hist(
        old_data,
        bins=60,
        weights=weights,
        range=(-500,500),
        histtype='step',
        label='All events, DNN. Mean: ' + str(round(old_mean_all, 4)) + '. RMS: ' + str(round(old_rms_all, 4)),
        density=True,
        color='blue')
    plt.hist(
        mmc_data,
        bins=60,
        weights=weights,
        range=(-500,500),
        histtype='step',
        label='All events, MMC. Mean: ' + str(round(mmc_mean_all, 4)) + '. RMS: ' + str(round(mmc_rms_all, 4)),
        density=True,
        color='purple')
    plt.xlim((-500,500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ Residual')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/mdn_resid_compare_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        old_data,
        bins=60,
        weights=weights,
        range=(-500,500),
        histtype='step',
        label='All events, DNN. Mean: ' + str(round(old_mean_all, 4)) + '. RMS: ' + str(round(old_rms_all, 4)),
        density=True)
    plt.hist(
        old_data[indices_1],
        bins=60,
        weights=weights[indices_1],
        range=(-500,500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$<4$, DNN. Mean: ' + str(round(old_mean_1, 4)) + '. RMS: ' + str(round(old_rms_1, 4)),
        density=True)
    plt.hist(
        old_data[indices_2],
        bins=60,
        weights=weights[indices_2],
        range=(-500,500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$>4$, DNN. Mean: ' + str(round(old_mean_2, 4)) + '. RMS: ' + str(round(old_rms_2, 4)),
        density=True)
    plt.xlim((-500,500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ Residual')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/mdn_old_resid_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(
        mmc_data,
        bins=60,
        weights=weights,
        range=(-500,500),
        histtype='step',
        label='All events, MMC. Mean: ' + str(round(mmc_mean_all, 4)) + '. RMS: ' + str(round(mmc_rms_all, 4)),
        density=True)
    plt.hist(
        mmc_data[indices_1],
        bins=60,
        weights=weights[indices_1],
        range=(-500,500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$<4$, MMC. Mean: ' + str(round(mmc_mean_1, 4)) + '. RMS: ' + str(round(mmc_rms_1, 4)),
        density=True)
    plt.hist(
        mmc_data[indices_2],
        bins=60,
        weights=weights[indices_2],
        range=(-500,500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$>4$, MMC. Mean: ' + str(round(mmc_mean_2, 4)) + '. RMS: ' + str(round(mmc_rms_2, 4)),
        density=True)
    plt.xlim((-500,500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ Residual')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/mdn_mmc_resid_' + label + '.pdf')
    plt.close(fig)
    
    mean_1 = np.mean(mus[indices_1])
    mean_2 = np.mean(mus[indices_2])
    mean_all = np.mean(mus)
    rms_1 = np.sqrt(np.mean((mus[indices_1] - mean_1) * (mus[indices_1] - mean_1)))
    rms_2 = np.sqrt(np.mean((mus[indices_2] - mean_2) * (mus[indices_2] - mean_2)))
    rms_all = np.sqrt(np.mean((mus - mean_all) * (mus - mean_all)))
    
    fig = plt.figure()
    plt.hist(
        mus,
        bins=80,
        weights=weights,
        range=(0,1500),
        histtype='step',
        label='All events. Mean: ' + str(round(mean_all, 4)) + '. RMS: ' + str(round(rms_all, 4)),
        density=True)
    plt.hist(
        mus[indices_1],
        bins=80,
        weights=weights[indices_1],
        range=(0,1500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$<4$. Mean: ' + str(round(mean_1, 4)) + '. RMS: ' + str(round(rms_1, 4)),
        density=True)
    plt.hist(
        mus[indices_2],
        bins=80,
        weights=weights[indices_2],
        range=(0,1500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$>4$. Mean: ' + str(round(mean_2, 4)) + '. RMS: ' + str(round(rms_2, 4)),
        density=True)
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/mdn_pred_mhh_by_sigma_range_' + label + '.pdf')
    plt.close(fig)
    
    mean_1 = np.mean(old_preds[indices_1])
    mean_2 = np.mean(old_preds[indices_2])
    mean_all = np.mean(old_preds)
    rms_1 = np.sqrt(np.mean((old_preds[indices_1] - mean_1) * (old_preds[indices_1] - mean_1)))
    rms_2 = np.sqrt(np.mean((old_preds[indices_2] - mean_2) * (old_preds[indices_2] - mean_2)))
    rms_all = np.sqrt(np.mean((old_preds - mean_all) * (old_preds - mean_all)))
    
    fig = plt.figure()
    plt.hist(
        old_preds,
        bins=80,
        weights=weights,
        range=(0,1500),
        histtype='step',
        label='All events. Mean: ' + str(round(mean_all, 4)) + '. RMS: ' + str(round(rms_all, 4)),
        density=True)
    plt.hist(
        old_preds[indices_1],
        bins=80,
        weights=weights[indices_1],
        range=(0,1500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$<4$. Mean: ' + str(round(mean_1, 4)) + '. RMS: ' + str(round(rms_1, 4)),
        density=True)
    plt.hist(
        old_preds[indices_2],
        bins=80,
        weights=weights[indices_2],
        range=(0,1500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$>4$. Mean: ' + str(round(mean_2, 4)) + '. RMS: ' + str(round(rms_2, 4)),
        density=True)
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/old_pred_mhh_by_sigma_range_' + label + '.pdf')
    plt.close(fig)

    mean_1 = np.mean(mmc_preds[indices_1])
    mean_2 = np.mean(mmc_preds[indices_2])
    mean_all = np.mean(mmc_preds)
    rms_1 = np.sqrt(np.mean((mmc_preds[indices_1] - mean_1) * (mmc_preds[indices_1] - mean_1)))
    rms_2 = np.sqrt(np.mean((mmc_preds[indices_2] - mean_2) * (mmc_preds[indices_2] - mean_2)))
    rms_all = np.sqrt(np.mean((mmc_preds - mean_all) * (mmc_preds - mean_all)))
    
    fig = plt.figure()
    plt.hist(
        mmc_preds,
        bins=80,
        weights=weights,
        range=(0,1500),
        histtype='step',
        label='All events. Mean: ' + str(round(mean_all, 4)) + '. RMS: ' + str(round(rms_all, 4)),
        density=True)
    plt.hist(
        mmc_preds[indices_1],
        bins=80,
        weights=weights[indices_1],
        range=(0,1500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$<4$. Mean: ' + str(round(mean_1, 4)) + '. RMS: ' + str(round(rms_1, 4)),
        density=True)
    plt.hist(
        mmc_preds[indices_2],
        bins=80,
        weights=weights[indices_2],
        range=(0,1500),
        histtype='step',
        label=r'$\sigma(m_{HH})$/Resolution$>4$. Mean: ' + str(round(mean_2, 4)) + '. RMS: ' + str(round(rms_2, 4)),
        density=True)
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/mmc_pred_mhh_by_sigma_range_' + label + '.pdf')
    plt.close(fig)

def sigma_plots(mus, sigmas, fold_1_array, label, mvis, indices_1 = None, indices_2 = None):
    truths = fold_1_array['universal_true_mhh']
    weights = fold_1_array['EventInfo___NominalAuxDyn']['evtweight'] * fold_1_array['fold_weight']
    
    trimmed_sigmas = sigmas[np.where(sigmas < 1000)]
    
    mean_sigma = np.mean(trimmed_sigmas)
    rms_sigma = np.sqrt(np.mean((trimmed_sigmas - mean_sigma) * (trimmed_sigmas - mean_sigma)))
    median_sigma = np.median(sigmas)
    
    # Split indices on sigma ranges
    if (indices_1 is None):
        indices_1 = np.where(sigmas < 1)
    if (indices_2 is None):
        indices_2 = np.where(sigmas > 1)
    
    fig = plt.figure()
    plt.hist(
        sigmas,
        bins=35,
        weights=weights,
        range=(0,70),
        label='Mean: ' + str(round(mean_sigma, 4)) + '. Median: ' + str(round(median_sigma, 4)) + '. RMS: ' + str(round(rms_sigma, 4)))
    """
    plt.hist(
        sigmas[indices_1],
        bins=30,
        weights=weights[indices_1],
        range=(0,6))
    plt.hist(
        sigmas[indices_2],
        bins=30,
        weights=weights[indices_2],
        range=(0,6))
    """
    plt.xlim((0,70))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\sigma(m_{HH})$ (GeV)')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/mdn_sigma_' + label + '.pdf')
    plt.close(fig)
    
    rel_sigmas = sigmas / mus
    trimmed_rel_sigmas = rel_sigmas[np.where(rel_sigmas < 100)]
    mean_rel_sigma = np.mean(trimmed_rel_sigmas)
    rms_rel_sigma = np.sqrt(np.mean((trimmed_rel_sigmas - mean_rel_sigma) * (trimmed_rel_sigmas - mean_rel_sigma)))
    median_rel_sigma = np.median(rel_sigmas)
    
    fig = plt.figure()
    plt.hist(
        rel_sigmas,
        bins=30,
        weights=weights,
        range=(0,0.6),
        label='Mean: ' + str(round(mean_rel_sigma, 4)) + '. Median: ' + str(round(median_rel_sigma, 4)) + '. RMS: ' + str(round(rms_rel_sigma, 4)))
    plt.xlim((0,0.6))
    plt.ylim(bottom=0)
    plt.xlabel(r'$\sigma(m_{HH})/m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small')
    fig.savefig('plots/mdn_rel_sigma_' + label + '.pdf')
    plt.close(fig)

    data = (truths - mus) / sigmas
    mean_1 = np.mean(data[indices_1])
    mean_2 = np.mean(data[indices_2])
    mean_all = np.mean(data)
    rms_1 = np.sqrt(np.mean((data[indices_1] - mean_1) * (data[indices_1] - mean_1)))
    rms_2 = np.sqrt(np.mean((data[indices_2] - mean_2) * (data[indices_2] - mean_2)))
    rms_all = np.sqrt(np.mean((data - mean_all) * (data - mean_all)))
    
    fig = plt.figure()
    plt.hist(
        data,
        bins=50,
        weights=weights,
        range=(-5,5),
        histtype='step',
        label='All events. Mean: ' + str(round(mean_all, 4)) + '. RMS: ' + str(round(rms_all, 4)),
        density=True)
    plt.hist(
        data[indices_1],
        bins=50,
        weights=weights[indices_1],
        range=(-5,5),
        histtype='step',
        label=r'Relative $\sigma<$Resolution. Mean: ' + str(round(mean_1, 4)) + '. RMS: ' + str(round(rms_1, 4)),
        density=True)
    plt.hist(
        data[indices_2],
        bins=50,
        weights=weights[indices_2],
        range=(-5,5),
        histtype='step',
        label=r'Relative $\sigma>$Resolution. Mean: ' + str(round(mean_2, 4)) + '. RMS: ' + str(round(rms_2, 4)),
        density=True)
    plt.xlim((-5,5))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ Residual / $\sigma$')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/mdn_resid_over_sigma_' + label + '.pdf')
    plt.close(fig)
    
    data = truths - mus
    mean_1 = np.mean(data[indices_1])
    mean_2 = np.mean(data[indices_2])
    mean_all = np.mean(data)
    rms_1 = np.sqrt(np.mean((data[indices_1] - mean_1) * (data[indices_1] - mean_1)))
    rms_2 = np.sqrt(np.mean((data[indices_2] - mean_2) * (data[indices_2] - mean_2)))
    rms_all = np.sqrt(np.mean((data - mean_all) * (data - mean_all)))
    
    fig = plt.figure()
    plt.hist(
        data,
        bins=50,
        weights=weights,
        range=(-500,500),
        histtype='step',
        label='All events. Mean: ' + str(round(mean_all, 4)) + '. RMS: ' + str(round(rms_all, 4)),
        density=True)
    plt.hist(
        data[indices_1],
        bins=50,
        weights=weights[indices_1],
        range=(-500,500),
        histtype='step',
        label=r'Relative $\sigma<$Resolution. Mean: ' + str(round(mean_1, 4)) + '. RMS: ' + str(round(rms_1, 4)),
        density=True)
    plt.hist(
        data[indices_2],
        bins=50,
        weights=weights[indices_2],
        range=(-500,500),
        histtype='step',
        label=r'Relative $\sigma>$Resolution. Mean: ' + str(round(mean_2, 4)) + '. RMS: ' + str(round(rms_2, 4)),
        density=True)
    plt.xlim((-500,500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$ Residual')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/mdn_resid_' + label + '.pdf')
    plt.close(fig)
    
    mean_1 = np.mean(truths[indices_1])
    mean_2 = np.mean(truths[indices_2])
    mean_all = np.mean(truths)
    rms_1 = np.sqrt(np.mean((truths[indices_1] - mean_1) * (truths[indices_1] - mean_1)))
    rms_2 = np.sqrt(np.mean((truths[indices_2] - mean_2) * (truths[indices_2] - mean_2)))
    rms_all = np.sqrt(np.mean((truths - mean_all) * (truths - mean_all)))
    
    fig = plt.figure()
    plt.hist(
        truths,
        bins=80,
        weights=weights,
        range=(0,1500),
        histtype='step',
        label='All events. Mean: ' + str(round(mean_all, 4)) + '. RMS: ' + str(round(rms_all, 4)),
        density=True)
    plt.hist(
        truths[indices_1],
        bins=80,
        weights=weights[indices_1],
        range=(0,1500),
        histtype='step',
        label=r'Relative $\sigma<$Resolution. Mean: ' + str(round(mean_1, 4)) + '. RMS: ' + str(round(rms_1, 4)),
        density=True)
    plt.hist(
        truths[indices_2],
        bins=80,
        weights=weights[indices_2],
        range=(0,1500),
        histtype='step',
        label=r'Relative $\sigma>$Resolution. Mean: ' + str(round(mean_2, 4)) + '. RMS: ' + str(round(rms_2, 4)),
        density=True)
    plt.xlim((0,1500))
    plt.ylim(bottom=0)
    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small')
    fig.savefig('plots/true_mhh_by_sigma_range_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist2d(x=mus, y=rel_sigmas, bins=[80,80], range=[[0,1500],[0,0.6]])
    plt.xlim((0,1500))
    plt.ylim((0,0.6))
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.ylabel(r'$\sigma (m_{HH})/m_{HH}$')
    fig.savefig('plots/rel_sigma_v_mhh_2d_hist_' + label + '.pdf')
    plt.close(fig)
    
    fig = plt.figure()
    plt.scatter(x=mus, y=rel_sigmas, s=1, marker='.')
    plt.xlim((0,1500))
    plt.ylim((0,0.1))
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.ylabel(r'$\sigma (m_{HH})/m_{HH}$')
    fig.savefig('plots/rel_sigma_v_mhh_scatter_low_' + label + '.pdf')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(x=mus, y=rel_sigmas, s=1, marker='.')
    plt.xlim((0,1500))
    plt.ylim((0.1,0.3))
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.ylabel(r'$\sigma (m_{HH})/m_{HH}$')
    fig.savefig('plots/rel_sigma_v_mhh_scatter_med_' + label + '.pdf')
    plt.close(fig)

    fig = plt.figure()
    plt.scatter(x=mus, y=rel_sigmas, s=1, marker='.')
    plt.xlim((0,1500))
    plt.ylim((0.3,0.8))
    plt.xlabel(r'$m_{HH}$ (GeV)')
    plt.ylabel(r'$\sigma (m_{HH})/m_{HH}$')
    fig.savefig('plots/rel_sigma_v_mhh_scatter_high_' + label + '.pdf')
    plt.close(fig)

def rnn_mmc_comparison(predictions_rnn, test_target, ak_array, ak_array_fold_1_array, label, regressor, predictions_old = None, predictions_mmc = None, sigma_label = '', sigma_slice = None):

    eff_tot = []
    fig = plt.figure()
    weights = ak_array_fold_1_array['EventInfo___NominalAuxDyn']['evtweight']*ak_array_fold_1_array['fold_weight']
    
    if sigma_slice is not None:
        weights = weights[sigma_slice]
        predictions_rnn = predictions_rnn[sigma_slice]
        test_target = test_target[sigma_slice]
        if predictions_mmc is not None:
            predictions_mmc = predictions_mmc[sigma_slice]
        if predictions_old is not None:
            predictions_old = predictions_old[sigma_slice]

    rms_rnn = sqrt(mean_squared_error(predictions_rnn, test_target))
    if predictions_mmc is not None:
        rms_mmc = sqrt(mean_squared_error(predictions_mmc, test_target))
    if predictions_old is not None:
        rms_old = sqrt(mean_squared_error(predictions_old, test_target))

    (n_rnn, bins_rnn, patches_rnn) = plt.hist(
        predictions_rnn,
        bins=80,
        weights=weights,
        range=(0,1500),
        #label=ak_array.title + '- RNN. Raw RMS: ' + str(round(rms_rnn, 4)) + '.',
        label=ak_array.title + '- MDN. Raw RMS: ' + str(round(rms_rnn, 4)) + '.',
        color='red',
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
        color='red',
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
            #label = ak_array.title + '- DNN. Raw RMS: ' + str(round(rms_mmc, 4)) + '.',
            color='purple',
            linestyle='solid',
            linewidth=2,
            histtype='step')

        eff_tot.append(calculate_eff(n_mmc))
        
    if predictions_old is not None:
        (n_old, bins_old, patches_old) = plt.hist(
            predictions_old,
            bins=80,
            weights= weights,
            range=(0,1500),
            #label = ak_array.title + '- MMC. Raw RMS: ' + str(round(rms_mmc, 4)) + '.',
            label = ak_array.title + '- DNN. Raw RMS: ' + str(round(rms_old, 4)) + '.',
            color='blue',
            linestyle='solid',
            linewidth=2,
            histtype='step')

    plt.xlabel(r'$m_{HH}$')
    plt.ylabel('Events')
    plt.ylim(bottom=0)
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + sigma_label + '_distributions_{}.pdf'.format(regressor))
    plt.close(fig)

    fig = plt.figure()

    (n_true, bins_true, patches_true) = plt.hist(
        test_target,
        bins=80,
        weights= weights,
        range=(0,1500),
        label=ak_array.title,
        color='red',
        linestyle='dashed',
        linewidth=2,
        histtype='step')

    eff_true = calculate_eff(n_true)

    plt.xlabel(r'True $m_{HH}$')
    plt.ylabel('Events')
    plt.legend(fontsize='small', numpoints=3)
    fig.savefig('plots/' + str(label) + sigma_label + '_distributions_truthonly.pdf')
    plt.close(fig)

    fig = plt.figure()

    ratio_rnn = predictions_rnn / test_target
    avg_ratio_rnn = np.average(ratio_rnn)
    target_ratio_rnn = [1.0] * len(ratio_rnn)
    rms_ratio_rnn = sqrt(mean_squared_error(ratio_rnn, target_ratio_rnn))

    if predictions_mmc is not None:
        ratio_mmc = predictions_mmc / test_target
        avg_ratio_mmc = np.average(ratio_mmc)
        target_ratio_mmc = [1.0] * len(ratio_mmc)
        rms_ratio_mmc = sqrt(mean_squared_error(ratio_mmc, target_ratio_mmc))
        
    if predictions_old is not None:
        ratio_old = predictions_old / test_target
        avg_ratio_old = np.average(ratio_old)
        target_ratio_old = [1.0] * len(ratio_old)
        rms_ratio_old = sqrt(mean_squared_error(ratio_old, target_ratio_old))

    (n_rat_rnn, bins_rat_rnn, patches_rat_rnn) = plt.hist(
        ratio_rnn,
        #label= ak_array.title + '- RNN. Raw Mean: ' + str(round(avg_ratio_rnn, 4)) + '. Raw RMS: ' + str(round(rms_ratio_rnn, 4)) + '.',
        label= ak_array.title + '- MDN. Raw Mean: ' + str(round(avg_ratio_rnn, 4)) + '. Raw RMS: ' + str(round(rms_ratio_rnn, 4)) + '.',
        color='red',
        weights= weights,
        bins=160,
        range=(0., 3.),
        linewidth=2,
        histtype='step')

    gauss_fit_calculator(n_rat_rnn, bins_rat_rnn, label, 'RNN', network_label = 'MDN')

    if predictions_mmc is not None:
        (n_rat_mmc, bins_rat_mmc, patches_rat_mmc) = plt.hist(
            ratio_mmc,
            label= ak_array.title + '- MMC. Raw Mean: ' + str(round(avg_ratio_mmc, 4)) + '. Raw RMS: ' + str(round(rms_ratio_mmc, 4)) + '.',
            #label= ak_array.title + '- DNN. Raw Mean: ' + str(round(avg_ratio_mmc, 4)) + '. Raw RMS: ' + str(round(rms_ratio_mmc, 4)) + '.',
            color='purple',
            weights= weights,
            bins=160,
            range=(0., 3.),
            linewidth=2,
            histtype='step')

        gauss_fit_calculator(n_rat_mmc, bins_rat_mmc, label, 'MMC')
        #gauss_fit_calculator(n_rat_mmc, bins_rat_mmc, label, 'RNN')
        
    if predictions_old is not None:
        (n_rat_old, bins_rat_old, patches_rat_old) = plt.hist(
            ratio_old,
            label= ak_array.title + '- DNN. Raw Mean: ' + str(round(avg_ratio_old, 4)) + '. Raw RMS: ' + str(round(rms_ratio_old, 4)) + '.',
            color='blue',
            weights= weights,
            bins=160,
            range=(0., 3.),
            linewidth=2,
            histtype='step')

        gauss_fit_calculator(n_rat_old, bins_rat_old, label, 'RNN', network_label = 'DNN')

    plt.xlabel(r'$m_{HH}$: prediction / truth')
    plt.ylabel('Events')
    plt.legend(fontsize='xx-small', numpoints=3)
    fig.savefig('plots/' + str(label) + sigma_label + '_ratios_{}.pdf'.format(regressor))
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


def roc_plot_rnn_mmc(eff, eff_true, name_1, name_2, sigma_label=''):

    auc_rnn = auc(eff[0], eff[2])
    auc_mmc = auc(eff[1], eff[3])
    auc_true = auc(eff_true[0], eff_true[1])

    fig = plt.figure()
    #plt.plot(eff[0], eff[2], color = 'red', label = 'RNN. AUC = ' + str(auc_rnn) + '.')
    plt.plot(eff[0], eff[2], color = 'red', label = 'MDN. AUC = ' + str(auc_rnn) + '.')
    plt.plot(eff[1], eff[3], color = 'purple', label = 'MMC. AUC = ' + str(auc_mmc) + '.')
    #plt.plot(eff[1], eff[3], color = 'green', label = 'Original RNN. AUC = ' + str(auc_mmc) + '.')
    plt.plot(eff_true[0], eff_true[1], color = 'green', label = 'Truth. AUC = ' + str(auc_true) + '.')
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
    fig.savefig('plots/' + title_1 + '_' + title_2 + '_mmc_rnn_roc' + sigma_label + '.pdf')
    plt.close(fig)


def nn_history(history, metric='loss'):

    fig = plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    #fig.axes[0].set_yscale('log')
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
