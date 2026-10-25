"""Golden-section search for a unimodal scalar function.

The golden-section search minimises a unimodal function f on a closed
interval [a, b] without requiring derivatives. At each step it evaluates
f at two interior points spaced according to the golden ratio, keeps the
triple that brackets the minimum, and shrinks the bracket by a factor of
about 0.618 per iteration.

The result is returned as a 3-tuple (x_min, f_min, iterations) so callers
can inspect both the location and the value, plus how many loop iterations
were actually performed (useful for sanity-checking convergence).
"""

from __future__ import annotations

from typing import Callable, Tuple

# Golden ratio conjugate: (sqrt(5) - 1) / 2 ≈ 0.6180339887.
# Using a precomputed float is cheaper than math.sqrt on every call and
# avoids importing math at all; the precision is well below float epsilon
# for any realistic iteration count.
_GOLDEN = 0.6180339887498949

def golden_section_search(
    func: Callable[[float], float],
    lower: float,
    upper: float,
    x_tolerance: float = 1.0e-10,
    max_iterations: int = 100,
) -> Tuple[float, float, int]:
    """Minimise a unimodal function on [lower, upper].

    Args:
        func: Unimodal scalar function to minimise.
        lower: Left endpoint of the bracket. Must be strictly less than upper.
        upper: Right endpoint of the bracket.
        x_tolerance: Stop once the bracket width is at or below this value.
        max_iterations: Hard cap on loop iterations to prevent runaway loops
            when the tolerance is too small for floating-point arithmetic
            to further reduce the bracket.

    Returns:
        A tuple (x_min, f_min, iterations) where x_min is the estimated
        minimiser, f_min is func(x_min), and iterations is the number of
        shrink steps actually performed.

    Raises:
        TypeError: if lower or upper is not a real number (int or float).
        ValueError: if lower >= upper.
        ValueError: if x_tolerance is not positive.
        ValueError: if max_iterations is not a positive integer.
    """
    # --- Input validation -------------------------------------------------
    # We reject non-real types early so callers get a clear message instead
    # of a confusing arithmetic TypeError deep in the loop. bool is a
    # subclass of int in Python; we reject it because a boolean endpoint is
    # almost certainly a caller bug, not an intent to use 0/1 as bounds.
    if isinstance(lower, bool) or not isinstance(lower, (int, float)):
        raise TypeError("lower must be a real number (int or float)")
    if isinstance(upper, bool) or not isinstance(upper, (int, float)):
        raise TypeError("upper must be a real number (int or float)")

    # bool is also rejected for x_tolerance for the same reason.
    if isinstance(x_tolerance, bool) or not isinstance(x_tolerance, (int, float)):
        raise TypeError("x_tolerance must be a real number (int or float)")

    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool):
        raise TypeError("max_iterations must be an int")

    if lower >= upper:
        raise ValueError(
            "lower must be strictly less than upper, got "
            f"lower={lower!r}, upper={upper!r}"
        )

    if x_tolerance <= 0:
        raise ValueError("x_tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")

    # --- Search -----------------------------------------------------------
    a = float(lower)
    b = float(upper)

    # Initial interior points. The golden ratio guarantees that each new
    # point reuses one of the previous evaluations, so after the first two
    # calls we only need one function evaluation per iteration.
    x_left = a + (1.0 - _GOLDEN) * (b - a)
    x_right = a + _GOLDEN * (b - a)

    f_left = float(func(x_left))
    f_right = float(func(x_right))

    iteration = 0
    while (b - a) > x_tolerance and iteration < max_iterations:
        iteration += 1

        if f_left < f_right:
            # The minimum lies in [a, x_right]. Shift the right point to
            # the old left point; we already know its function value.
            b = x_right
            x_right = x_left
            f_right = f_left
            x_left = a + (1.0 - _GOLDEN) * (b - a)
            f_left = float(func(x_left))
        else:
            # The minimum lies in [x_left, b]. Shift the left point to the
            # old right point; we already know its function value.
            a = x_left
            x_left = x_right
            f_left = f_right
            x_right = a + _GOLDEN * (b - a)
            f_right = float(func(x_right))

    # Report the better of the two interior points. When the loop exits
    # because of max_iterations, the bracket may still be wide; using the
    # better of the two current samples is a safe, deterministic choice
    # that requires no extra function evaluations.
    if f_left <= f_right:
        return x_left, f_left, iteration
    return x_right, f_right, iteration
