import math

def calculate_square_root(number):
    """
    This function calculates the square root of a given number.
    """
    if number < 0:
        raise ValueError("Cannot calculate the square root of a negative number.")
    return math.sqrt(number)
