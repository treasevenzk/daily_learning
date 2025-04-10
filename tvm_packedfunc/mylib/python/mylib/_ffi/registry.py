"""Global function registry."""

from . import function

_GLOBAL_FUNC_CACHE = {}

def get_global_func(name):
    if name in _GLOBAL_FUNC_CACHE:
        return _GLOBAL_FUNC_CACHE[name]
    
    func = function.get_global_func(name)
    if func is not None:
        _GLOBAL_FUNC_CACHE[name] = func
    return func