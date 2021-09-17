import os
import json
import awkward as ak
import numpy as np
import h5py
from bbtautau import log; log = log.getChild(__file__)

def _array_to_hdf5(ak_array, file_name):
    """
    """
    h5file = h5py.File(file_name, "w")
    group = h5file.create_group("awkward")
    form, length, container = ak.to_buffers(ak_array, container=group)
    group.attrs["form"] = form.tojson()
    group.attrs["length"] = json.dumps(length)

if __name__ == '__main__':

    from bbtautau.database import dihiggs_01, dihiggs_10, ztautau, ttbar

    # signal
    for sample in [
            dihiggs_01,
            dihiggs_10,
    ]:
        log.info('building cache for {}'.format(sample.name))
        sample.process(use_cache=False, run_mmc=True)
        _file_name = os.path.join(
            'cache',
            '{}.h5'.format(sample.name))
        _array_to_hdf5(sample.ak_array, _file_name)

    # background
    for sample in [
            ztautau,
            # ttbar,
    ]:

        log.info('building cache for {}'.format(sample.name))
        sample.process(is_signal=False, use_cache=False, run_mmc=True)
        _file_name = os.path.join(
            'cache',
            '{}.h5'.format(sample.name))
        _array_to_hdf5(sample.ak_array, _file_name)
