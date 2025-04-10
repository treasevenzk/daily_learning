"""Example showing how to use PackedFun from Python."""

import packedfun
from packedfun._ffi import get_global_func, list_global_func

def main():
    """Main function to demonstrate PackedFun usage."""
    # List available functions
    print("Available functions:")
    for func_name in list_global_func():
        print(f"- {func_name}")
    print()
    
    # Get functions
    add_func = get_global_func("AddIntegers")
    greet_func = get_global_func("Greet")
    
    if add_func is None:
        print("AddIntegers function not found!")
    else:
        # Call AddIntegers
        result = add_func(5, 7)
        print(f"5 + 7 = {result}")
    
    if greet_func is None:
        print("Greet function not found!")
    else:
        # Call Greet
        result = greet_func("PackedFun User")
        print(result)

if __name__ == "__main__":
    main()