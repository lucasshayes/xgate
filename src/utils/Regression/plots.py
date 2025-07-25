import numpy as np
import matplotlib.pyplot as plt


def plot_ecdf_with_percentiles(
    sorted_errors: np.ndarray,
    ecdf_values: np.ndarray,
    dir: str,
    percentiles: list[float] = [0.5, 0.99],
    title: str = "ECDF with Percentiles",
) -> None:
    """
    Plot ECDF from precomputed sorted errors and ECDF values, mark specified percentiles,
    and optionally save the plot to a file.

    Args:
        sorted_errors (np.ndarray): 1D sorted array of errors.
        ecdf_values (np.ndarray): 1D array of ECDF values corresponding to sorted_errors.
        dir (str): Directory to save plot file.
        percentiles (list[float]): Percentiles to mark on the plot (values between 0 and 1).
        title (str): Plot title.
    """

    # Input validation
    if sorted_errors.ndim != 1 or ecdf_values.ndim != 1:
        raise ValueError("sorted_errors and ecdf_values must be 1D arrays.")
    if len(sorted_errors) != len(ecdf_values):
        raise ValueError("sorted_errors and ecdf_values must be the same length.")
    for p in percentiles:
        if not (0 < p < 1):
            raise ValueError(f"Percentiles must be between 0 and 1. Got {p}.")

    # Create base plot
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_errors, ecdf_values, marker=".", linestyle="none", label="ECDF")
    plt.xlabel("Error")
    plt.ylabel("ECDF (Cumulative Probability)")
    plt.title(title)

    n = len(sorted_errors)

    # For each percentile
    for i, p in enumerate(percentiles):
        # Get intersect index
        idx = np.searchsorted(ecdf_values, p, side="left")
        idx = min(idx, n - 1)
        p_error = sorted_errors[idx]

        # Draw percentile lines
        plt.hlines(
            y=p,
            xmin=0,
            xmax=p_error,
            colors="r",
            linestyles="dashed",
            # Label for legend area
            label=f"p{int(p * 100)} = {p_error:.3f}" if i == 0 else None,
        )
        plt.vlines(p_error, ymin=0, ymax=p, colors="r", linestyles="dashed")

        # Annotation offset
        x_offset = p_error * 0.05 if p_error > 0 else 0.05
        y_offset = -0.05

        # Annotate plot
        plt.annotate(
            f"p{int(p * 100)}={p_error:.3f}",
            xy=(p_error, p),
            xytext=(p_error + x_offset, p + y_offset),
            # arrowprops=dict(facecolor="black", arrowstyle="->"),
            fontsize=10,
        )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(dir)
    plt.close()
