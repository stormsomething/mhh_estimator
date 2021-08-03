"""
"""
# list the branches of interests
EVT_FIELDS = [
    'EventInfo___NominalAuxDyn.MCEventWeight',
    'EventInfo___NominalAuxDyn.eventNumber',
    'EventInfo___NominalAuxDyn.mcChannelNumber',
    'EventInfo___NominalAuxDyn.CorrectedAndScaledAvgMu',

    ]
TRUTH_FIELDS = [
    # truth
    'TruthParticles___NominalAuxDyn.pdgId',
    'TruthParticles___NominalAuxDyn.px',
    'TruthParticles___NominalAuxDyn.py',
    'TruthParticles___NominalAuxDyn.pz',
    'TruthParticles___NominalAuxDyn.e',
    'TruthParticles___NominalAuxDyn.status'
    ]
TAU_FIELDS = [
    # taus
    'TauJets___NominalAuxDyn.pt',
    'TauJets___NominalAuxDyn.eta',
    'TauJets___NominalAuxDyn.phi',
    'TauJets___NominalAuxDyn.m',
    'TauJets___NominalAuxDyn.TATTruthMatch',
    'TauJets___NominalAuxDyn.isRNNMedium',
    'TauJets___NominalAuxDyn.nTracks',
    ]
BJETS_FIELDS = [
    # bjets
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.pt',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.eta',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.phi',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.m',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.isBJet',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.PartonTruthLabelID',
    'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn.TruthPartonLabelID',
    ]
MET_FIELDS = [
    # MET
    'MET_Reference_AntiKt4EMPFlow___NominalAuxDyn.mpx',
    'MET_Reference_AntiKt4EMPFlow___NominalAuxDyn.mpy',
    'MET_Reference_AntiKt4EMPFlow___NominalAuxDyn.sumet',
    'MET_Reference_AntiKt4EMPFlow___NominalAuxDyn.metSig',
]

FIELDS = EVT_FIELDS + TRUTH_FIELDS + TAU_FIELDS + BJETS_FIELDS + MET_FIELDS
