# tiny_golden_section_search

A minimal, dependency-free implementation of the golden-section search for minimising a unimodal scalar function on a closed interval, with no derivative required.

```python
from tiny_golden_section_search import golden_section_search

x_min, f_min, iterations = golden_section_search(
    lambda x: (x - 3.0) ** 2,
    lower=-10.0,
    upper=10.0,
)
# x_min ≈ 3.0, f_min ≈ 0.0
```

## Why this exists

Some optimisation problems hand you a unimodal objective whose derivative is unavailable, expensive, or noisy. Golden-section search shrinks a bracket by a constant factor (~0.618) per iteration and only needs one new function evaluation per step after the first. The trade-off: it is linearly convergent, so it is slower than methods that use gradients — but it asks for nothing beyond the function value itself.

## Edge cases you will hit

The function must be unimodal on the interval; if it is not, the result is whatever subinterval the algorithm happened to lock onto, with no warning. For a flat function the tie-break prefers the left branch, so the bracket drifts toward the left endpoint. Inputs are validated: `lower` must be strictly less than `upper`, `x_tolerance` must be positive, `max_iterations` must be a positive integer, and bool endpoints are rejected (they are almost always a caller bug, despite being a subclass of `int` in Python). The search returns `(x_min, f_min, iterations)` so you can inspect both the location and how many shrink steps ran.

## Exports

`golden_section_search(func, lower, upper, x_tolerance=1.0e-10, max_iterations=100) -> (x_min, f_min, iterations)`
