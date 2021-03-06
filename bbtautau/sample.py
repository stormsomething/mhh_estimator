import uproot
import os

from .fields import FIELDS
class sample(object):

    def __init__(
            self,
            name,
            title,
            color,
            dsid, 
            path,
            tree='CollectionTree'):

        self._name = name
        self._title = title
        self._color = color
        if isinstance(dsid, (list, tuple)):
            self._dsids = dsid
        else:
            self._dsids = [dsid]
        self._path = path
        self._tree = tree
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
        if self._ak_array == None:
            self._open()
        return self._ak_array
        
    @property
    def fold_0_array(self):
        return self._fold_0_array

    @property
    def fold_1_array(self):
        return self._fold_1_array

    def _open(self, max_files=None):
        # build file list
        _paths = []
        for _dsid in self._dsids:
            for _dir in os.listdir(self._path):
                # if not a directory, skip
                if not os.path.isdir(os.path.join(
                        self._path, _dir)):
                    continue
                if str(_dsid) in _dir:
                    for _file in os.listdir(os.path.join(
                            self._path, _dir)):
                        _fullpath = os.path.join(
                            self._path, _dir, _file)
                        _fullpath += ':' + self._tree
                        _paths.append(_fullpath)
        if max_files != None:
            if not isinstance(max_files, int):
                raise ValueError
            _paths = _paths[:max_files]
            # _fullpath = self._path + '/*' + str(_dsid) + '*/*.root:' + self._tree
        # _paths =             _fullpath = self._path + '/*' + str(_dsid) + '/*.root:' + self._tree
        self._ak_array = uproot.concatenate(_paths, FIELDS, how='zip')
                                       
    def process(self, max_files=None, **kwargs):
        from .selector import _select
        if self._ak_array == None:
            self._open(max_files=max_files)
        self._ak_array = _select(self._ak_array, **kwargs)
        from .utils import train_test_split
        self._fold_0_array, self._fold_1_array = train_test_split(self._ak_array)
