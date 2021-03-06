import os
from .sample import sample

DEFAULT_PATH = '/Users/quentin/bbtautau_mhh/data/'
PATH = os.getenv("CXAOD_PATH", DEFAULT_PATH)
dihiggs_01 = sample('HH_01', 'HH 01', 'red', 600023, PATH)
dihiggs_10 = sample('HH_10', 'HH 10', 'blue', 600024, PATH)
