"""Test PackedFun function calls from Python."""

import unittest
import packedfun
from packedfun._ffi import get_global_func, list_global_func

class TestPackedFun(unittest.TestCase):
    """Test cases for PackedFun Python bindings."""
    
    def test_list_functions(self):
        """Test listing global functions."""
        func_names = list_global_func()
        self.assertIsInstance(func_names, list)
        # The actual functions will depend on what was registered
        
    def test_add_integers(self):
        """Test the AddIntegers function."""
        add_func = get_global_func("AddIntegers")
        self.assertIsNotNone(add_func)
        
        result = add_func(10, 20)
        self.assertEqual(result, 30)
        
    def test_greet(self):
        """Test the Greet function."""
        greet_func = get_global_func("Greet")
        self.assertIsNotNone(greet_func)
        
        result = greet_func("Python Tester")
        self.assertEqual(result, "Hello, Python Tester!")
        
    def test_nonexistent_function(self):
        """Test getting a function that doesn't exist."""
        func = get_global_func("NonExistentFunction")
        self.assertIsNone(func)

if __name__ == "__main__":
    unittest.main()