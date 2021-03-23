# import cpp as cpp_module
from . import cpp_module

for _name in cpp_module.__all__:
    _func = getattr(cpp_module, _name)
    if callable(_func):
        _func()
