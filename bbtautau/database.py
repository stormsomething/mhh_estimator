import os
from .sample import sample

DEFAULT_PATH = '/Users/quentin/bbtautau_mhh/data/'
PATH = os.getenv("CXAOD_PATH", DEFAULT_PATH)
dihiggs_01 = sample(
    'HH_01',
    r'$HH \to bb\tau_{h}\tau_{h} (\kappa_{\lambda} = 1)$',
    # r'$HH (\lambda = 1.)$',
    'red',
    600023, PATH)
dihiggs_10 = sample(
    'HH_10', r'$HH \to bb \tau_{h}\tau_{h} (\kappa_{\lambda} = 10)$', 'blue',
    600024, PATH)
