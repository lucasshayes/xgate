import numpy as np


def ecdf(errors: list) -> tuple:
    """
    Compute the Empirical Cumulative Distribution Function (ECDF) of the given errors.
    (For final evaluation)

    Args:
        errors (list): A list of error values.

    Returns:
        tuple: A tuple containing the sorted error values and their corresponding ECDF values.
    """
    # Sort the errors
    sorted_errors = np.sort(errors)
    # Compute the ECDF values
    ecdf_values = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    return sorted_errors, ecdf_values


def p_ecdf(errors: list, p: float) -> float:
    """
    Compute the p-th percentile of the ECDF of the given errors.
    (For final evaluation)

    Args:
        errors (list): A list of error values.
        p (float): The percentile to compute (between 0 and 1).

    Returns:
        float: The p-th percentile value.
    """
    # Exceptions
    if not (0 <= p <= 1):
        raise ValueError("Percentile p must be between 0 and 1.")
    if not errors:
        raise ValueError("Error list cannot be empty.")

    # Compute ecdf
    sorted_errors, ecdf_values = ecdf(errors)
    # Find the index where the ECDF is greater than or equal to p
    index = np.searchsorted(ecdf_values, p)

    return sorted_errors[index] if index < len(sorted_errors) else sorted_errors[-1]
