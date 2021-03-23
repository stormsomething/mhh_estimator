__all__ = [
    '_load_mmc_cpp',
    '_load_mmc_runner_cpp',
    ]
  

import ROOT


from .. import log
log = log.getChild(__name__)

def _load_mmc_cpp():
    try:
        ROOT.gSystem.CompileMacro('bbtautau/cpp/standaloneMMC/MissingMassCalculatorV2.cxx', 'k-', '', 'cache')
        log.info('mmc loaded!')
    except:
        raise RuntimeError

def _load_mmc_runner_cpp():
    try:
        ROOT.gSystem.CompileMacro('bbtautau/cpp/mmc_runner.cpp', 'k-', '', 'cache')
        log.info('mmc runner loaded!')
    except:
        raise RuntimeError

#    try:
#        ROOT.VBF_Tagger()
#    except:
#        print "VBF Tagger not compiled"
