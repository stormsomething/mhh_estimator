import awkward as ak
import numpy as np
from itertools import product

def true_mhh(ak_array):
    # return the two higgs
    higgs_pairs = ak.combinations(ak_array.higgs, 2)
    h1, h2 = ak.unzip(higgs_pairs)
    # compute the target m_hh and flatten it
    _true_mhh =  np.sqrt((h1.e + h2.e)**2 - (h1.px + h2.px)**2 - (h1.py + h2.py)**2 - (h1.pz + h2.pz)**2) / 1000.
    return _true_mhh

# can change this to give a custom scaling factor to use as the target
def m_ratio(ak_array):
    higgs_pairs = ak.combinations(ak_array.higgs, 2)
    h1, h2 = ak.unzip(higgs_pairs)
    _true_mhh =  np.sqrt((h1.e + h2.e)**2 - (h1.px + h2.px)**2 - (h1.py + h2.py)**2 - (h1.pz + h2.pz)**2) / 1000.
    _transv_m = np.sqrt((h1.e + h2.e)**2 - (h1.pz + h2.pz)**2) / 1000
    ratio = _true_mhh / _transv_m
    return ratio

def transv_m(ak_array):
    higgs_pairs = ak.combinations(ak_array.higgs, 2)
    h1, h2 = ak.unzip(higgs_pairs)
    _transv_m = np.sqrt((h1.e + h2.e)**2 - (h1.pz + h2.pz)**2) / 1000
    return _transv_m

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
        #ak_array['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['metSig'],
        #ak_array['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['metSig_PU'],
    ], axis=1)
    table = ak.to_numpy(table)
    return table


def train_test_split(ak_array, modulus=3):
    _train = ak_array[ak_array['EventInfo___NominalAuxDyn.eventNumber'] % modulus != 0]
    _test  = ak_array[ak_array['EventInfo___NominalAuxDyn.eventNumber'] % modulus == 0]
    return _train, _test
