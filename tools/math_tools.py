from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> float:
    """Adds two numbers and returns the result rounded to 2 decimal places.

    Args:
        a (float): First number to be added
        b (float): Second number to be added

    Returns:
        float: The sum of a and b, rounded to 2 decimal places
    """
    return round((a + b), 2)


@tool
def sub(a: float, b: float) -> float:
    """Subtracts the second number from the first and returns the result rounded to 2 decimal places.

    Args:
        a (float): Number to subtract from
        b (float): Number to subtract

    Returns:
        float: The difference between a and b, rounded to 2 decimal places
    """
    return round((a - b), 2)


@tool
def mult(a: float, b: float) -> float:
    """Multiplies two numbers and returns the result rounded to 2 decimal places.

    Args:
        a (float): First number to multiply
        b (float): Second number to multiply

    Returns:
        float: The product of a and b, rounded to 2 decimal places
    """
    return round((a * b), 2)


@tool
def div(a: float, b: float) -> float:
    """Divides the first number by the second and returns the result rounded to 2 decimal places.

    Args:
        a (float): Number to be divided (dividend)
        b (float): Number to divide by (divisor)

    Raises:
        ValueError: If the divisor (b) is zero

    Returns:
        float: The quotient of a divided by b, rounded to 2 decimal places
    """
    if b == 0:
        raise ValueError("Cannot divide by zero!")

    return round((a / b), 2)


@tool
def mod(a: float, b: float) -> float:
    """Calculates the remainder of dividing the first number by the second and returns the result rounded to 2 decimal places.

    Args:
        a (float): Number to be divided (dividend)
        b (float): Number to divide by (divisor)

    Returns:
        float: The remainder of a divided by b, rounded to 2 decimal places
    """
    return round((a % b), 2)
