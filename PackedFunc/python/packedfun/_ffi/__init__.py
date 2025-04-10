"""FFI package for PackedFun."""

from .base import PackedFunError
from .function import PackedFunction
from .registry import get_global_func, register_func, list_global_func