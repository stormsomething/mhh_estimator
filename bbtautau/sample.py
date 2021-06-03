import uproot
import awkward as ak
import os

from .fields import *
from .luminosity import LUMI
from . import log; log = log.getChild(__name__)

_XSEC_FILTER_KFAC = {
    600023: {'xsec': 0.027887, 'filter': 0.14537, 'kfactor': 1.},
    600024: {'xsec': 0.58383, 'filter': 0.13638, 'kfactor': 1.},
    364128: {'xsec': 1981.7, 'filter': 8.3449E-01, 'kfactor': 0.9751},
    364129: {'xsec': 1981.7, 'filter': 1.0956E-01, 'kfactor': 0.9751},
    364130: {'xsec': 1982.1, 'filter': 6.5757E-02, 'kfactor': 0.9751},
    364131: {'xsec': 110.61, 'filter': 6.9266E-01, 'kfactor': 0.9751},
    364132: {'xsec': 110.46, 'filter': 1.9060E-01, 'kfactor': 0.9751},
    364133: {'xsec': 110.66, 'filter': 0.110886, 'kfactor': 0.9751},
    364134: {'xsec': 40.756, 'filter': 6.1880E-01, 'kfactor': 0.9751},
    364135: {'xsec': 40.716, 'filter': 2.3429E-01, 'kfactor': 0.9751},
    364136: {'xsec': 40.746, 'filter': 1.5530E-01, 'kfactor': 0.9751},
    364137: {'xsec': 8.6639, 'filter': 5.6340E-01, 'kfactor': 0.9751},
    364138: {'xsec': 8.676, 'filter': 2.6433E-01, 'kfactor': 0.9751},
    364139: {'xsec': 8.6795, 'filter': 1.7627E-01, 'kfactor': 0.9751},
    364140: {'xsec': 1.8078, 'filter': 1., 'kfactor': 0.9751},
    364141: {'xsec': 0.14826, 'filter': 1., 'kfactor': 0.9751},
}
    


class sample(object):

    def __init__(
            self,
            name,
            title,
            color,
            dsid, 
            path,
            tree='CollectionTree',
            metadata='MetaData_EventCount'):

        self._name = name
        self._title = title
        self._color = color
        if isinstance(dsid, (list, tuple)):
            self._dsids = dsid
        else:
            self._dsids = [dsid]
        self._path = path
        self._tree = tree
        self._metadata = metadata

        self._ak_array = None
        self._fold_0_array = None
        self._fold_1_array = None

    @property
    def name(self):
        return self._name

    @property
    def title(self):
        return self._title

    @property
    def color(self):
        return self._color
    
    @property
    def ak_array(self):
        if not isinstance(self._ak_array, ak.Array):
            self._open()
        return self._ak_array
        
    @property
    def fold_0_array(self):
        return self._fold_0_array

    @property
    def fold_1_array(self):
        return self._fold_1_array


    # @property
    # def total_pred(self):
    #     return self._xsec * 1000. * self._filt * self._kfac * LUMI

    def _open(self, max_files=None):
        # build file list
        _paths = {}
        for _dsid in self._dsids:
            _paths_tree = []
            _paths_meta = []
            for _dir in os.listdir(self._path):
                # if not a directory, skip
                if not os.path.isdir(os.path.join(
                        self._path, _dir)):
                    continue
                if not str(_dsid) in _dir:
                    continue
                for _file in os.listdir(os.path.join(
                        self._path, _dir)):
                    _fullpath = os.path.join(
                        self._path, _dir, _file)
                    _fullpath_tree = _fullpath + ':' + self._tree
                    _fullpath_meta = _fullpath + ':' + self._metadata
                    _paths_tree.append(_fullpath_tree)
                    _paths_meta.append(_fullpath_meta)
            _paths[_dsid] = {
                'tree': _paths_tree,
                'meta': _paths_meta,
                }

        # opening a limited amount of files
        # in debug mode
        if max_files != None:
            if not isinstance(max_files, int):
                raise ValueError
            for _dsid in self._dsids:
                _paths[_dsid]['tree'] = _paths[_dsid]['tree'][:max_files]
                _paths[_dsid]['meta'] = _paths[_dsid]['meta'][:max_files]
                _paths_meta = _paths_meta[:max_files]

        log.info('computing sum-of-weights for sample {}'.format(self.name))
        _sow = {}
        for _dsid in self._dsids:
            _sow_dsid = []
            for _path in _paths[_dsid]['meta']:
                _hist = uproot.open(_path)
                _idx = _hist.axis().labels().index('sumOfWeights initial')
                _sow_dsid.append(_hist.values()[_idx])
            _sow[_dsid] = sum(_sow_dsid)
            log.info('\t {}: s-o-w = {}'.format(
                _dsid, _sow[_dsid]))
        log.info('sample {}, using files:'.format(self.name))
        for _dsid in self._dsids:
            for _f in _paths[_dsid]['tree']:
                 log.info('\t' + _f)
        # use uproot.concatenate (for now)
        for _dsid in self._dsids:
            log.info('adding ' + str(_dsid))
            _ak_array = uproot.concatenate(
                    _paths[_dsid]['tree'], EVT_FIELDS, how='zip', num_workers=4)
            _ak_array_truth = uproot.concatenate(
                    _paths[_dsid]['tree'], TRUTH_FIELDS, how='zip', num_workers=4)
            _ak_array_tau = uproot.concatenate(
                    _paths[_dsid]['tree'], TAU_FIELDS, how='zip', num_workers=4)
            _ak_array_bjets = uproot.concatenate(
                    _paths[_dsid]['tree'], BJETS_FIELDS, how='zip', num_workers=4)
            _ak_array_met = uproot.concatenate(
                    _paths[_dsid]['tree'], MET_FIELDS, how='zip', num_workers=4)
            _ak_array['TauJets___NominalAuxDyn'] = _ak_array_tau['TauJets___NominalAuxDyn']
            _ak_array['TruthParticles___NominalAuxDyn'] = _ak_array_truth['TruthParticles___NominalAuxDyn']
            _ak_array['AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn'] = _ak_array_bjets['AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn']
            _ak_array['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn'] = _ak_array_met['MET_Reference_AntiKt4EMPFlow___NominalAuxDyn']
            _pred  = _XSEC_FILTER_KFAC[_dsid]['xsec'] * 1000. # xsec in fb
            _pred *= _XSEC_FILTER_KFAC[_dsid]['filter']
            _pred *= _XSEC_FILTER_KFAC[_dsid]['kfactor']
            _pred *= LUMI
            _pred /= _sow[_dsid]
            _ak_array['evtweight'] = _ak_array['EventInfo___NominalAuxDyn.MCEventWeight'] * _pred
            if not isinstance(self._ak_array, ak.Array):
                self._ak_array = _ak_array
            else:
                self._ak_array = ak.concatenate([self._ak_array, _ak_array])
                
    def process(self, max_files=None, **kwargs):
        from .selector import _select
        if self._ak_array == None:
            self._open(max_files=max_files)
        self._ak_array = _select(self._ak_array, **kwargs)
        from .utils import train_test_split
        self._fold_0_array, self._fold_1_array = train_test_split(self._ak_array)
