import uproot
import awkward as ak
import numpy as np
import joblib
from itertools import product

# rfile = uproot.open('../group.phys-hdbs.21766918._000005.CxAOD.root')
# tree = rfile['CollectionTree']


# list the branches of interests
fields = [
    'EventInfo___NominalAuxDyn.eventNumber',
    # truth
    'TruthParticles___NominalAuxDyn.pdgId',
    'TruthParticles___NominalAuxDyn.px',
    'TruthParticles___NominalAuxDyn.py',
    'TruthParticles___NominalAuxDyn.pz',
    'TruthParticles___NominalAuxDyn.e',
    # taus
    'TauJets___NominalAuxDyn.pt',
    'TauJets___NominalAuxDyn.eta',
    'TauJets___NominalAuxDyn.phi',
    'TauJets___NominalAuxDyn.m',
    'TauJets___NominalAuxDyn.TATTruthMatch',
    'TauJets___NominalAuxDyn.isRNNMedium',
    'TauJets___NominalAuxDyn.nTracks',
    # bjets
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.pt',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.eta',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.phi',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.m',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.isBJet',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.PartonTruthLabelID',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.TruthPartonLabelID',
    # MET
    'MET_Reference_AntiKt4EMPFlow___NominalAuxDyn.mpx',
    'MET_Reference_AntiKt4EMPFlow___NominalAuxDyn.mpy',
    'MET_Reference_AntiKt4EMPFlow___NominalAuxDyn.sumet',
]


print 'start'
# load the data as a zipped awkward array
# events = uproot.concatenate('../data/group.phys-hdbs.mc16_13TeV.600024.HIGG4D3.e7954_s3126_r10724_p3978.bbtt_hh_v15-01_CxAOD.root/*.root:CollectionTree', fields, how='zip')
events = uproot.concatenate('../data/*600023*/*.root:CollectionTree', fields, how='zip')
print 'opened!'

print len(events)

# build the higgs object (subset of truth particles)
events['higgs'] = events.TruthParticles___NominalAuxDyn[events.TruthParticles___NominalAuxDyn.pdgId == 25]
events = events[ak.num(events.higgs) == 2]
print len(events)

# build the selected taus container (simple truth matching selection)
events['taus'] = events.TauJets___NominalAuxDyn[events.TauJets___NominalAuxDyn.TATTruthMatch == 15]

# print len(events)
events = events[ak.num(events.taus) == 2]
print len(events)

# apply tau ID
events['taus'] = events.taus[events.taus.isRNNMedium == 1]
events = events[ak.num(events.taus) == 2]
print len(events)

# build the selected b-jets container (simple truth matching selection)
events['bjets'] = events.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn[events.AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.isBJet == 1]
events = events[ak.num(events.bjets) == 2]
print len(events)

# return the two higgs 
higgs_pairs = ak.combinations(events.higgs, 2)
h1, h2 = ak.unzip(higgs_pairs)

# compute the target m_hh and flatten it
true_mhh =  np.sqrt((h1.e + h2.e)**2 - (h1.px + h2.px)**2 - (h1.py + h2.py)**2 - (h1.pz + h2.pz)**2) / 1000.
# true_pthh = np.sqrt((h1.px + h2.px)**2 + (h1.py + h2.py)**2) / 1000.

train_events = events[events['EventInfo___NominalAuxDyn.eventNumber'] % 3 != 0]
test_events = events[events['EventInfo___NominalAuxDyn.eventNumber'] % 3 == 0]
print len(train_events), len(test_events)

train_target = ak.flatten(true_mhh[events['EventInfo___NominalAuxDyn.eventNumber'] % 3 != 0])
test_target = ak.flatten(true_mhh[events['EventInfo___NominalAuxDyn.eventNumber'] % 3 == 0])

def features_table(ak_array):
    # build a rectilinear table for training using  pt, eta, phi, m of the taus and b-jets
    table = ak.concatenate([
        ak_array[obj][var][:, idx, None]
        for obj, var, idx in product(["taus", "bjets"], ["pt", "eta", "phi", "m"], range(2))
    ], axis=1)
    # adding the MET
    table = ak.concatenate([
        table,
        ak_array['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['mpx'],
        ak_array['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']['mpy'],
    ], axis=1)
    table = ak.to_numpy(table)
    return table


    
from sklearn.ensemble import GradientBoostingRegressor
# feature optimisation
# from sklearn.model_selection import train_test_split
# features_train, features_test, target_train, target_test = train_test_split(
#     table, true_mhh, test_size=0.33, random_state=42)
# from sklearn.model_selection import GridSearchCV
# parameters = {
#     # 'n_estimators': [100, 200, 400, 600, 800, 1000],
#     'n_estimators': [5000, 10000, 20000],
#     'learning_rate': [0.1],
#     'max_depth': [5],
#     'loss': ['ls'],
#     }
# gbr = GradientBoostingRegressor()
# regressor = GridSearchCV(gbr, parameters, n_jobs=4)
# best_regressor = regressor.best_estimator_
# joblib.dump(best_regressor, 'training/best.clf')
# add a plot?

fit = False
if fit == True:
    best_regressor = GradientBoostingRegressor(
        n_estimators=5000,
        learning_rate=0.1,
        max_depth=5,
        random_state=0,
        loss='ls')
    features_train = features_table(train_events)
    print 'shape', features_train.shape
    best_regressor.fit(features_train, train_target)
    joblib.dump(best_regressor, 'weights/best.clf')

else:
    best_regressor = joblib.load('weights/best.clf')

features_test = features_table(test_events)
predictions = best_regressor.predict(features_test)

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
plt.hist(ak.flatten(events['taus']['nTracks']))
plt.xlabel('nTracks')
fig.savefig('plots/nTracks.pdf')
plt.close(fig)

fig = plt.figure()
plt.hist(ak.flatten(events['taus']['isRNNMedium']))
plt.xlabel('RNN Medium ID Decision')
fig.savefig('plots/rnn_medium_id.pdf')
plt.close(fig)

