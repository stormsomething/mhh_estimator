import uproot
import awkward as ak
import h5py
import json
import numpy as np
import os
#import ROOT

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
    410470: {'xsec': 729.77, 'filter': 5.4384E-01, 'kfactor': 1.13975636159},
    410471: {'xsec': 729.78, 'filter': 4.5627E-01, 'kfactor': 1.13974074379},
    410644: {'xsec': 2.027, 'filter': 1., 'kfactor': 1.0170},
    410645: {'xsec': 1.2674, 'filter': 1., 'kfactor': 1.0167},
    410646: {'xsec': 37.936, 'filter': 1., 'kfactor': 0.9450},
    410647: {'xsec': 37.905, 'filter': 1., 'kfactor': 0.9457},
    410658: {'xsec': 36.996, 'filter': 1., 'kfactor': 1.1935},
    410659: {'xsec': 22.175, 'filter': 1., 'kfactor': 1.1849},
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
        _ak_arrays = []
        #_watch = ROOT.TStopwatch()
        for _dsid in self._dsids:
            log.info('adding ' + str(_dsid))
            #_watch.Print()
            #_watch.Start()
            _ak_array = uproot.concatenate(_paths[_dsid]['tree'], filter_name=lambda l: l in FIELDS)
            #_watch.Print()
            #_watch.Start()
            _pred  = _XSEC_FILTER_KFAC[_dsid]['xsec'] * 1000. # xsec in fb
            _pred *= _XSEC_FILTER_KFAC[_dsid]['filter']
            _pred *= _XSEC_FILTER_KFAC[_dsid]['kfactor']
            _pred *= LUMI
            _pred /= _sow[_dsid]

            _ak_evt = ak.zip({f.split('.')[-1]: _ak_array[f] for f in EVT_FIELDS})
            _ak_evt['evtweight'] = _ak_array['EventInfo___NominalAuxDyn.MCEventWeight'] * _pred

            _ak_truth = ak.zip({f.split('.')[-1]: _ak_array[f] for f in TRUTH_FIELDS})
            _ak_tau = ak.zip({f.split('.')[-1]: _ak_array[f] for f in TAU_FIELDS})
            _ak_bjets = ak.zip({f.split('.')[-1]: _ak_array[f] for f in BJETS_FIELDS})
            _ak_met = ak.zip({f.split('.')[-1]: _ak_array[f] for f in MET_FIELDS})

            _ak_array = ak.zip({
                'EventInfo___NominalAuxDyn': _ak_evt,
                'TruthParticles___NominalAuxDyn': _ak_truth,
                'TauJets___NominalAuxDyn': _ak_tau,
                'AntiKt4EMPFlowJets_BTagging201903___NominalAuxDyn': _ak_bjets,
                'MET_Reference_AntiKt4EMPFlow___NominalAuxDyn': _ak_met,
            }, depth_limit=1)
            _ak_arrays += [_ak_array]

        log.info('concatenating into a single array')
        if not isinstance(self._ak_array, ak.Array):
            self._ak_array = ak.concatenate(_ak_arrays)


    def process(self, max_files=None, use_cache=False, remove_bad_training_events=False, run_mmc=False, **kwargs):
        if use_cache:
            self._ak_array = self.load_from_cache()

        if not isinstance(self._ak_array, ak.Array):
            self._open(max_files=max_files)

            from .selector import _select
            self._ak_array = _select(self._ak_array, **kwargs)

            from .utils import universal_true_mhh
            _mhh = universal_true_mhh(self._ak_array, self._name)
            self._ak_array['universal_true_mhh'] = ak.from_numpy(universal_true_mhh(self._ak_array, self._name))

        if run_mmc:
            if not 'mmc' in self._ak_array.fields:
                from .mmc import mmc
                _mmc, _mhh_mmc = mmc(self._ak_array)
                self._ak_array['mmc_tautau'] = _mmc
                self._ak_array['mmc_bbtautau'] = _mhh_mmc

        if remove_bad_training_events:
            n_before_cleaning = len(self._ak_array)
            self._ak_array = self._ak_array[self._ak_array['universal_true_mhh'] > -1000]
            log.info('{} -- Events with well defined true mhh: {}/{}'.format(
                self._name, len(self._ak_array), n_before_cleaning))
            
        from .utils import train_test_split
        self._fold_0_array, self._fold_1_array = train_test_split(self._ak_array)

    def load_from_cache(self):
        log.warning('{} -- loading awkward array from cache!'.format(self._name))
        log.warning('{} -- use at your own risk!'.format(self._name))

        _file_name = os.path.join(
            'cache',
            '{}.h5'.format(self._name))
        if not os.path.exists(_file_name):
            log.error('{} does not exist!'.format(_file_name))
            return None

        h5file = h5py.File(_file_name)
        group = h5file['awkward']
        reconstituted = ak.from_buffers(
            ak.forms.Form.fromjson(group.attrs["form"]),
            json.loads(group.attrs["length"]),
            {k: np.asarray(v) for k, v in group.items()},
        )
        return reconstituted
