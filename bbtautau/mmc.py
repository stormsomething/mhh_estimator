# import numba as nb
import numpy as np
import ROOT
import bbtautau.cpp

# @nb.jit
def mmc(ak_array):
    out_mmc = np.empty(len(ak_array), np.float64)
    out_mhh = np.empty(len(ak_array), np.float64)
    for ievt, evt in enumerate(ak_array):
        mmc_val = ROOT.HackedMMC.MMC(
            evt['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['mpx'][0] / 1000.,
            evt['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['mpy'][0] / 1000.,
            evt['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['sumet'][0] / 1000.,
            2,
            evt['taus']['pt'][0] / 1000.,
            evt['taus']['eta'][0],
            evt['taus']['phi'][0],
            evt['taus']['pt'][1] / 1000.,
            evt['taus']['eta'][1],
            evt['taus']['phi'][1],
            evt['EventInfo___NominalAuxDyn.eventNumber'])
        b_1 = ROOT.TLorentzVector()
        b_1.SetPtEtaPhiM(
            evt['bjets']['pt'][0] / 1000.,
            evt['bjets']['eta'][0],
            evt['bjets']['phi'][0],
            evt['bjets']['m'][0] / 1000.)
        b_2 = ROOT.TLorentzVector()
        b_2.SetPtEtaPhiM(
            evt['bjets']['pt'][1] / 1000.,
            evt['bjets']['eta'][1],
            evt['bjets']['phi'][1],
            evt['bjets']['m'][1] / 1000.)
        out_mmc[ievt] = mmc_val.M()
        out_mhh[ievt] = (mmc_val + b_1 + b_2).M()
    return out_mmc, out_mhh
