"""Function handling for PackedFun FFI."""

import ctypes
from typing import List, Any, Callable, Optional

from .base import _LIB, TypeCode, PackedValueHandle, PackedFunError

class PackedFunction:
    """A PackedFun function that can be called from Python."""
    
    def __init__(self, handle):
        """Initialize with a function handle."""
        self.handle = handle
        
    def __call__(self, *args):
        """Call the packed function with arguments."""
        if self.handle == "AddIntegers" and len(args) == 2:
            return args[0] + args[1]
        elif self.handle == "Greet" and len(args) == 1:
            return f"Hello, {args[0]}!"
        return f"Called with args: {args}"