import awkward as ak
import numpy as np
import math
import ROOT
from itertools import product
import scipy.stats as sc
import matplotlib.pyplot as plt


def universal_true_mhh(ak_array, name):

    # truth information
    truth_pdgId = ak_array.TruthParticles___NominalAuxDyn['pdgId']
    truth_status = ak_array.TruthParticles___NominalAuxDyn['status']
    truth_px = ak_array.TruthParticles___NominalAuxDyn['px']
    truth_py = ak_array.TruthParticles___NominalAuxDyn['py']
    truth_pz = ak_array.TruthParticles___NominalAuxDyn['pz']
    truth_e = ak_array.TruthParticles___NominalAuxDyn['e']

    # reconstructed b-jets
    reco_isbjet = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['isBJet']
    reco_pt = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['pt']
    reco_phi = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['phi']
    reco_eta = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['eta']
    reco_m = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['m']

    # form the four-momenta for reco b-jets
    reco_px = reco_pt*np.cos(reco_phi)
    reco_py = reco_pt*np.sin(reco_phi)
    reco_pz = reco_pt*np.sinh(reco_eta)
    reco_e = np.sqrt(reco_m**2 + reco_px**2 + reco_py**2 + reco_pz**2)

    incorrect_count = 0
    # deletions = []

    universal_truth_mhh = []
    if (name == 'HH_01' or name == 'HH_10' or name == 'ttbar'):
        for i in range(len(truth_pdgId)):
            px_tot = 0
            py_tot = 0
            pz_tot = 0
            e_tot = 0

            status = truth_status[i].tolist()
            pdgId = truth_pdgId[i].tolist()

            # for the taus
            tau_indices = [y for y, x in enumerate(pdgId) if (x == 15 or x == -15)]
            count_taus = len(tau_indices)
            if (count_taus > 2):
                taus_status = []
                for k in range(len(status)):
                    if k in tau_indices:
                        taus_status.append(status[k])
                final_taus_indices = []
                for j in range(2):
                    minimum = min(taus_status)
                    min_index = taus_status.index(minimum)
                    real_min_index = tau_indices[min_index]
                    final_taus_indices.append(real_min_index)
                    taus_status.remove(minimum)
                    tau_indices.remove(real_min_index)
                px_tot = px_tot + truth_px[i][final_taus_indices[0]] + truth_px[i][final_taus_indices[1]]
                py_tot = py_tot + truth_py[i][final_taus_indices[0]] + truth_py[i][final_taus_indices[1]]
                pz_tot = pz_tot + truth_pz[i][final_taus_indices[0]] + truth_pz[i][final_taus_indices[1]]
                e_tot = e_tot + truth_e[i][final_taus_indices[0]] + truth_e[i][final_taus_indices[1]]
                count_taus = 2

            else: # if count_taus <= 2
                for j in range(len(truth_pdgId[i])):
                    if (truth_pdgId[i][j] == 15 or truth_pdgId[i][j] == -15):
                        px_tot = px_tot + truth_px[i][j]
                        py_tot = py_tot + truth_py[i][j]
                        pz_tot = pz_tot + truth_pz[i][j]
                        e_tot = e_tot + truth_e[i][j]

            # for the b-jets
            bjet_indices = [y for y, x in enumerate(pdgId) if (x == 5 or x == -5)]
            count_bjets = len(bjet_indices)
            if (count_bjets < 2):
                temp_count_bjets = 0
                for j in range(len(reco_isbjet[i])):
                    if (reco_isbjet[i][j] == 1):
                        px_tot = px_tot + reco_px[i][j]
                        py_tot = py_tot + reco_py[i][j]
                        pz_tot = pz_tot + reco_pz[i][j]
                        e_tot = e_tot + reco_e[i][j]
                        temp_count_bjets = temp_count_bjets + 1
                count_bjets = temp_count_bjets
            else: # if count_bjets >= 2
                for j in range(len(truth_pdgId[i])):
                    if (truth_pdgId[i][j] == 5 or truth_pdgId[i][j] == -5):
                        px_tot = px_tot + truth_px[i][j]
                        py_tot = py_tot + truth_py[i][j]
                        pz_tot = pz_tot + truth_pz[i][j]
                        e_tot = e_tot + truth_e[i][j]
            if (count_taus != 2 or count_bjets != 2):
                print('Not the right number of particles! -- ' + str(name) + '. Taus: ' + str(count_taus) + '. Bjets: ' + str(count_bjets))
                incorrect_count = incorrect_count + 1
                mhh = -1000
            else:
                mhh = math.sqrt(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2) / 1000.
            universal_truth_mhh.append(mhh)

    elif (name == 'ztautau'):
        for i in range(len(truth_pdgId)):
            px_tot = 0
            py_tot = 0
            pz_tot = 0
            e_tot = 0
            count_taus = 0
            count_bjets = 0
            # for the taus
            for j in range(len(truth_pdgId[i])):
                if (truth_pdgId[i][j] == 15 or truth_pdgId[i][j] == -15):
                    px_tot = px_tot + truth_px[i][j]
                    py_tot = py_tot + truth_py[i][j]
                    pz_tot = pz_tot + truth_pz[i][j]
                    e_tot = e_tot + truth_e[i][j]
                    count_taus = count_taus + 1
            # for the b-jets (need to use reco, unfortunately)
            for k in range(len(reco_isbjet[i])):
                if (reco_isbjet[i][k] == 1):
                    px_tot = px_tot + reco_px[i][k]
                    py_tot = py_tot + reco_py[i][k]
                    pz_tot = pz_tot + reco_pz[i][k]
                    e_tot = e_tot + reco_e[i][k]
                    count_bjets = count_bjets + 1
            if (count_taus != 2 or count_bjets != 2):
                print('Not the right number of particles! -- ' + str(name) + '. Taus: ' + str(count_taus) + '. Bjets: ' + str(count_bjets))
                incorrect_count = incorrect_count + 1
                mhh = -1000
            else:
                mhh = math.sqrt(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2) / 1000.
            universal_truth_mhh.append(mhh)
    else:
        raise ValueError('wrong sample name!')

    print('Number of events in sample ' + str(name) + ' without exactly 2 taus and 2 bjets: ' + str(incorrect_count) + '.')
    return np.array(universal_truth_mhh)


def visable_mass(ak_array, name):

    pdgId = ak_array.TruthParticles___NominalAuxDyn['pdgId']

    # reconstructed b-jets
    bjet_reco_isbjet = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['isBJet']
    bjet_reco_pt = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['pt']
    bjet_reco_phi = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['phi']
    bjet_reco_eta = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['eta']
    bjet_reco_m = ak_array.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn['m']

    # reconstructed taus
    tau_reco_istau = ak_array.TauJets___NominalAuxDyn['isRNNMedium']
    tau_reco_pt = ak_array.TauJets___NominalAuxDyn['pt']
    tau_reco_phi = ak_array.TauJets___NominalAuxDyn['phi']
    tau_reco_eta = ak_array.TauJets___NominalAuxDyn['eta']
    tau_reco_m = ak_array.TauJets___NominalAuxDyn['m']

    # form the four-momenta for reco b-jets
    bjet_reco_px = bjet_reco_pt*np.cos(bjet_reco_phi)
    bjet_reco_py = bjet_reco_pt*np.sin(bjet_reco_phi)
    bjet_reco_pz = bjet_reco_pt*np.sinh(bjet_reco_eta)
    bjet_reco_e = np.sqrt(bjet_reco_m**2 + bjet_reco_px**2 + bjet_reco_py**2 + bjet_reco_pz**2)

    # form the four-momenta for reco taus
    tau_reco_px = tau_reco_pt*np.cos(tau_reco_phi)
    tau_reco_py = tau_reco_pt*np.sin(tau_reco_phi)
    tau_reco_pz = tau_reco_pt*np.sinh(tau_reco_eta)
    tau_reco_e = np.sqrt(tau_reco_m**2 + tau_reco_px**2 + tau_reco_py**2 + tau_reco_pz**2)

    incorrect_count = 0

    vis_mass = []
    for i in range(len(pdgId)):
        px_tot = 0
        py_tot = 0
        pz_tot = 0
        e_tot = 0
        count_taus = 0
        count_bjets = 0
        # for the b-jets
        for j in range(len(bjet_reco_isbjet[i])):
            if (bjet_reco_isbjet[i][j] == 1):
                px_tot = px_tot + bjet_reco_px[i][j]
                py_tot = py_tot + bjet_reco_py[i][j]
                pz_tot = pz_tot + bjet_reco_pz[i][j]
                e_tot = e_tot + bjet_reco_e[i][j]
                count_bjets = count_bjets + 1
        # for the taus
        for k in range(len(tau_reco_istau[i])):
            if (tau_reco_istau[i][k] == 1 and count_taus < 2):
                px_tot = px_tot + tau_reco_px[i][k]
                py_tot = py_tot + tau_reco_py[i][k]
                pz_tot = pz_tot + tau_reco_pz[i][k]
                e_tot = e_tot + tau_reco_e[i][k]
                count_taus = count_taus + 1
        m = math.sqrt(e_tot**2 - px_tot**2 - py_tot**2 - pz_tot**2) / 1000.
        vis_mass.append(m)

    return np.array(vis_mass)


def features_table(ak_array):
    #  b-jets
    table = ak.concatenate([
        ak_array['bjets'][var][:, idx, None]
        for var, idx in product(["pt", "eta", "phi", "m"], range(2))
    ], axis=1)
    # taus
    table = ak.concatenate([table] + [
        ak_array['taus'][var][:, idx, None]
        for var, idx in product(["pt", "eta", "phi"], range(2))
    ], axis=1)
    # adding the MET
    table = ak.concatenate([
        table,
        ak_array['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['mpx'],
        ak_array['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['mpy'],
        ak_array['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['metSig']
    ], axis=1)
    table = ak.to_numpy(table)
    return table


def clean_samples(fold_array, deletions):

    fold_array_new = fold_array.tolist()
    bad_entires = [x for i, x in enumerate(fold_array_new) if i in deletions]
    final_fold_array = []
    for y in fold_array_new:
        if y not in bad_entires:
            final_fold_array.append(y)
    final_fold_array = ak.Array(final_fold_array)

    return final_fold_array


def train_test_split(ak_array, modulus=3):
    _train = ak_array[ak_array['EventInfo___NominalAuxDyn']['eventNumber'] % modulus != 0]
    _test  = ak_array[ak_array['EventInfo___NominalAuxDyn']['eventNumber'] % modulus == 0]
    _train['fold_weight'] = float(modulus) / float(modulus - 1)
    _test['fold_weight'] = float(modulus)
    return _train, _test


def gauss_fit_calculator(n, bins, name, fitter):

    mids = 0.5*(bins[1:] + bins[:-1])
    mids_new = []
    n_new = []
    for i in range(len(mids)):
        if (name == 'dihiggs_01' and fitter == 'RNN'):
            if (mids[i] >= 0.6 and mids[i] <= 1.3):
                mids_new.append(mids[i])
                n_new.append(n[i])
        elif (name == 'dihiggs_01' and fitter == 'MMC'):
            if (mids[i] >= 0.55 and mids[i] <= 1.2):
                mids_new.append(mids[i])
                n_new.append(n[i])
        elif (name == 'dihiggs_10' and fitter == 'RNN'):
            if (mids[i] >= 0.6 and mids[i] <= 1.3):
                mids_new.append(mids[i])
                n_new.append(n[i])
        elif (name == 'dihiggs_10' and fitter == 'MMC'):
            if (mids[i] >= 0.55 and mids[i] <= 1.3):
                mids_new.append(mids[i])
                n_new.append(n[i])
        elif (name == 'ztautau' and fitter == 'RNN'):
            if (mids[i] >= 0.8 and mids[i] <= 1.3):
                mids_new.append(mids[i])
                n_new.append(n[i])
        elif (name == 'ztautau' and fitter == 'MMC'):
            if (mids[i] >= 0.8 and mids[i] <= 1.2):
                mids_new.append(mids[i])
                n_new.append(n[i])
        elif (name == 'ttbar' and fitter == 'RNN'):
            if (mids[i] >= 0.6 and mids[i] <= 1.4):
                mids_new.append(mids[i])
                n_new.append(n[i])
        elif (name == 'ttbar' and fitter == 'MMC'):
            if (mids[i] >= 0.6 and mids[i] <= 1.4):
                mids_new.append(mids[i])
                n_new.append(n[i])
    mids = np.array(mids_new)
    n = np.array(n_new)
    height = max(n)
    mu = np.average(mids, weights=n)
    sigma = np.sqrt(np.average((mids - mu)**2, weights=n))
    x = np.linspace(0, 3, 160)
    if (fitter == 'RNN'):
        y = sc.norm.pdf(x, mu, sigma)
    elif (fitter == 'MMC' and (name == 'ttbar' or name == 'ztautau')):
        y = sc.norm.pdf(x, mu, sigma)
    else:
        max_index = np.argmax(n)
        mu = mids[max_index]
        y = sc.norm.pdf(x, mu, sigma)
    y_max = max(y)
    mult = height / y_max
    if (fitter == 'RNN'):
        plt.plot(x, mult*y, color='black', label= 'Gaussian Fit -- RNN. Mu: ' + str(round(mu, 4)) + '. Sigma: ' + str(round(sigma, 4)) + '.')
    elif (fitter == 'MMC'):
        plt.plot(x, mult*y, color='gray', label= 'Gaussian Fit -- MMC. Mu: ' + str(round(mu, 4)) + '. Sigma: ' + str(round(sigma, 4)) + '.')


def chi_square_test(predictions_HH_01, predictions_HH_10, predictions_ztautau, predictions_ttbar, dihiggs_01_fold_1_array, dihiggs_10_fold_1_array, ztautau_fold_1_array, ttbar_fold_1_array, fitter, mult_HH_01, mult_HH_10, mult_ztautau=1, mult_ttbar=1):

    weights_HH_01 = dihiggs_01_fold_1_array['EventInfo___NominalAuxDyn']['evtweight']*dihiggs_01_fold_1_array['fold_weight']*mult_HH_01
    weights_HH_10 = dihiggs_10_fold_1_array['EventInfo___NominalAuxDyn']['evtweight']*dihiggs_10_fold_1_array['fold_weight']*mult_HH_10
    weights_ztautau = ztautau_fold_1_array['EventInfo___NominalAuxDyn']['evtweight']*ztautau_fold_1_array['fold_weight']*mult_ztautau
    weights_ttbar = ttbar_fold_1_array['EventInfo___NominalAuxDyn']['evtweight']*ttbar_fold_1_array['fold_weight']*mult_ttbar

    predictions_HH_01_bkg = np.concatenate([predictions_HH_01, predictions_ztautau, predictions_ttbar])
    predictions_HH_10_bkg = np.concatenate([predictions_HH_10, predictions_ztautau, predictions_ttbar])

    weights_HH_01_bkg = np.concatenate([weights_HH_01, weights_ztautau, weights_ttbar])
    weights_HH_10_bkg = np.concatenate([weights_HH_10, weights_ztautau, weights_ttbar])

    HH_01_bkg_hist = ROOT.TH1F('Statistics', 'Signal (HH_01) with Top Quark and Z -> tautau Background', 30, 200, 800)
    for i in range(len(predictions_HH_01_bkg)):
        HH_01_bkg_hist.Fill(predictions_HH_01_bkg[i], weights_HH_01_bkg[i])

    HH_10_bkg_hist = ROOT.TH1F('Statistics', 'Signal (HH_10) with Top Quark and Z -> tautau Background', 30, 200, 800)
    for j in range(len(predictions_HH_10_bkg)):
        HH_10_bkg_hist.Fill(predictions_HH_10_bkg[j], weights_HH_10_bkg[j])

    p_value = HH_01_bkg_hist.Chi2Test(HH_10_bkg_hist, option='WW')
    z_score = ROOT.RooStats.PValueToSignificance(p_value)
    print(fitter + ' p-value: ' + str(p_value) + '. ' + fitter + ' z-score: ' + str(z_score) + '.')

    return HH_01_bkg_hist, HH_10_bkg_hist


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The below functions are not currently in use, but they can be employed in order to attempt to further optimize the training.

def rotate_events(features):

    mpx = features[:,14]
    mpy = features[:,15]

    angles = np.arctan2(mpy, mpx)

    mpt = np.sqrt(mpx**2 + mpy**2)
    features[:,14] = mpt

    features_4 = features[:,4] - angles
    features_4_new = []
    for i in features_4:
        if i < -math.pi:
            features_4_new.append(i + 2*math.pi)
        elif i > math.pi:
            features_4_new.append(i - 2*math.pi)
        else:
            features_4_new.append(i)
    features[:,4] = np.array(features_4_new)

    features_5 = features[:,5] - angles
    features_5_new = []
    for j in features_5:
        if j < -math.pi:
            features_5_new.append(j + 2*math.pi)
        elif j > math.pi:
            features_5_new.append(j - 2*math.pi)
        else:
            features_5_new.append(j)
    features[:,5] = np.array(features_5_new)

    features_12 = features[:,12] - angles
    features_12_new = []
    for k in features_12:
        if k < -math.pi:
            features_12_new.append(k + 2*math.pi)
        elif k > math.pi:
            features_12_new.append(k - 2*math.pi)
        else:
            features_12_new.append(k)
    features[:,12] = np.array(features_12_new)

    features_13 = features[:,13] - angles
    features_13_new = []
    for l in features_13:
        if l < -math.pi:
            features_13_new.append(l + 2*math.pi)
        elif k > math.pi:
            features_13_new.append(l - 2*math.pi)
        else:
            features_13_new.append(l)
    features[:,13] = np.array(features_13_new)

    features = np.delete(features,15,1)

    return features


def true_mhh(ak_array):
    # return the two higgs
    higgs_pairs = ak.combinations(ak_array.higgs, 2)
    h1, h2 = ak.unzip(higgs_pairs)
    # compute the target m_hh and flatten it
    _true_mhh =  np.sqrt((h1.e + h2.e)**2 - (h1.px + h2.px)**2 - (h1.py + h2.py)**2 - (h1.pz + h2.pz)**2) / 1000.
    return _true_mhh


def create_weights(data, weights):

    count = [0]*20
    for i in range(len(data)):
        if (data[i] >= 0 and data[i] < 75):
            count[0] = count[0] + 1
        elif (data[i] >= 75 and data[i] < 150):
            count[1] = count[1] + 1
        elif (data[i] >= 150 and data[i] < 225):
            count[2] = count[2] + 1
        elif (data[i] >= 225 and data[i] < 300):
            count[3] = count[3] + 1
        elif (data[i] >= 300 and data[i] < 375):
            count[4] = count[4] + 1
        elif (data[i] >= 375 and data[i] < 450):
            count[5] = count[5] + 1
        elif (data[i] >= 450 and data[i] < 525):
            count[6] = count[6] + 1
        elif (data[i] >= 525 and data[i] < 600):
            count[7] = count[7] + 1
        elif (data[i] >= 600 and data[i] < 675):
            count[8] = count[8] + 1
        elif (data[i] >= 675 and data[i] < 750):
            count[9] = count[9] + 1
        elif (data[i] >= 750 and data[i] < 825):
            count[10] = count[10] + 1
        elif (data[i] >= 825 and data[i] < 900):
            count[11] = count[11] + 1
        elif (data[i] <= 900 and data[i] < 975):
            count[12] = count[12] + 1
        elif (data[i] <= 975 and data[i] < 1050):
            count[13] = count[13] + 1
        elif (data[i] <= 1050 and data[i] < 1125):
            count[14] = count[14] + 1
        elif (data[i] <= 1125 and data[i] < 1200):
            count[15] = count[15] + 1
        elif (data[i] <= 1200 and data[i] < 1275):
            count[16] = count[16] + 1
        elif (data[i] <= 1275 and data[i] < 1350):
            count[17] = count[17] + 1
        elif (data[i] <= 1350 and data[i] < 1425):
            count[18] = count[18] + 1
        else:
            count[19] = count[19] + 1

    flat = np.array([25]*20)
    count = np.array(count)
    for i in range(len(count)):
        if count[i] == 0:
            count[i] = 1
    adhoc = flat / count

    sample_weights = []
    for i in range(len(data)):
        if (data[i] >= 0 and data[i] < 75):
            sample_weights.append(float(weights[i]) * adhoc[0])
        elif (data[i] >= 75 and data[i] < 150):
            sample_weights.append(float(weights[i]) * adhoc[1])
        elif (data[i] >= 150 and data[i] < 225):
            sample_weights.append(float(weights[i]) * adhoc[2])
        elif (data[i] >= 225 and data[i] < 300):
            sample_weights.append(float(weights[i]) * adhoc[3])
        elif (data[i] >= 300 and data[i] < 375):
            sample_weights.append(float(weights[i]) * adhoc[4])
        elif (data[i] >= 375 and data[i] < 450):
            sample_weights.append(float(weights[i]) * adhoc[5])
        elif (data[i] >= 450 and data[i] < 525):
            sample_weights.append(float(weights[i]) * adhoc[6])
        elif (data[i] >= 525 and data[i] < 600):
            sample_weights.append(float(weights[i]) * adhoc[7])
        elif (data[i] >= 600 and data[i] < 675):
            sample_weights.append(float(weights[i]) * adhoc[8])
        elif (data[i] >= 675 and data[i] < 750):
            sample_weights.append(float(weights[i]) * adhoc[9])
        elif (data[i] >= 750 and data[i] < 825):
            sample_weights.append(float(weights[i]) * adhoc[10])
        elif (data[i] >= 825 and data[i] < 900):
            sample_weights.append(float(weights[i]) * adhoc[11])
        elif (data[i] <= 900 and data[i] < 975):
            sample_weights.append(float(weights[i]) * adhoc[12])
        elif (data[i] <= 975 and data[i] < 1050):
            sample_weights.append(float(weights[i]) * adhoc[13])
        elif (data[i] <= 1050 and data[i] < 1125):
            sample_weights.append(float(weights[i]) * adhoc[14])
        elif (data[i] <= 1125 and data[i] < 1200):
            sample_weights.append(float(weights[i]) * adhoc[15])
        elif (data[i] <= 1200 and data[i] < 1275):
            sample_weights.append(float(weights[i]) * adhoc[16])
        elif (data[i] <= 1275 and data[i] < 1350):
            sample_weights.append(float(weights[i]) * adhoc[17])
        elif (data[i] <= 1350 and data[i] < 1425):
            sample_weights.append(float(weights[i]) * adhoc[18])
        else:
            sample_weights.append(float(weights[i]) * adhoc[19])

    return sample_weights
