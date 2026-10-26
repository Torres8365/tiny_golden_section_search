"""Tests for the golden-section search implementation.

Tests are deterministic: they use pure functions with closed-form minima
and compare floats with a tolerance via assertAlmostEqual rather than ==.
No wall-clock timing, no sleeps, no randomness.
"""

import math
import unittest

from tiny_golden_section_search import golden_section_search


class TestGoldenSectionSearchHappyPath(unittest.TestCase):
    """Basic convergence on simple unimodal functions."""

    def test_minimises_simple_quadratic(self):
        # f(x) = (x - 3)^2 has minimum at x = 3.
        result = golden_section_search(lambda x: (x - 3.0) ** 2, -10.0, 10.0)
        x_min, f_min, iterations = result
        self.assertAlmostEqual(x_min, 3.0, places=7)
        self.assertAlmostEqual(f_min, 0.0, places=7)
        self.assertGreater(iterations, 0)

    def test_minimises_shifted_parabola(self):
        # f(x) = 5*(x+2)^2 + 1 has minimum at x = -2, value 1.
        result = golden_section_search(lambda x: 5.0 * (x + 2.0) ** 2 + 1.0, -5.0, 5.0)
        x_min, f_min, iterations = result
        self.assertAlmostEqual(x_min, -2.0, places=7)
        self.assertAlmostEqual(f_min, 1.0, places=7)
        self.assertGreater(iterations, 0)

    def test_minimises_cosine(self):
        # On [0, 2*pi], cos has its minimum at pi.
        result = golden_section_search(math.cos, 0.0, 2.0 * math.pi)
        x_min, f_min, iterations = result
        self.assertAlmostEqual(x_min, math.pi, places=6)
        self.assertAlmostEqual(f_min, -1.0, places=6)
        self.assertGreater(iterations, 0)


class TestGoldenSectionSearchReturnValue(unittest.TestCase):
    """The return value's type and structure."""

    def test_returns_three_tuple(self):
        result = golden_section_search(lambda x: x ** 2, -1.0, 1.0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_third_element_is_int_iterations(self):
        result = golden_section_search(lambda x: x ** 2, -1.0, 1.0)
        self.assertIsInstance(result[2], int)


class TestGoldenSectionSearchReturnValuesAgreeWithEvaluation(unittest.TestCase):
    """The returned f_min must equal a fresh evaluation at x_min."""

    def test_reported_value_matches_fresh_evaluation(self):
        # Use a function whose minimum is not exactly representable so we
        # exercise the real comparison path.
        target = 1.7
        func = lambda x: (x - target) ** 2 + 0.5
        x_min, f_min, _ = golden_section_search(func, 0.0, 5.0)
        self.assertAlmostEqual(f_min, func(x_min), places=12)


class TestGoldenSectionSearchParameters(unittest.TestCase):
    """Behaviour with explicit parameter choices."""

    def test_custom_tolerance_fewer_iterations(self):
        # A looser tolerance should require fewer iterations.
        loose = golden_section_search(lambda x: x ** 2, -1.0, 1.0, x_tolerance=1.0e-3)
        tight = golden_section_search(lambda x: x ** 2, -1.0, 1.0, x_tolerance=1.0e-10)
        self.assertLessEqual(loose[2], tight[2])

    def test_custom_max_iterations_caps_loop(self):
        # With max_iterations=5 the loop must stop early, regardless of the
        # tiny tolerance that would otherwise require many more steps.
        x_min, f_min, iterations = golden_section_search(
            lambda x: x ** 2, -1.0, 1.0, x_tolerance=1.0e-20, max_iterations=5
        )
        self.assertLessEqual(iterations, 5)
        # The returned x_min should still be within the original bracket.
        self.assertGreaterEqual(x_min, -1.0)
        self.assertLessEqual(x_min, 1.0)

    def test_int_endpoints_accepted(self):
        # Integer bounds should be accepted and produce correct results.
        x_min, f_min, _ = golden_section_search(lambda x: (x - 2) ** 2, 0, 5)
        self.assertAlmostEqual(x_min, 2.0, places=7)
        self.assertAlmostEqual(f_min, 0.0, places=7)


class TestGoldenSectionSearchEdgeCases(unittest.TestCase):
    """Awkward inputs that must be handled cleanly."""

    def test_min_at_left_endpoint(self):
        # f decreasing on [0, 1], minimum at 0. The search converges toward
        # the left edge but always keeps interior points, so x_min
        # approaches 0 without ever equaling it exactly.
        x_min, f_min, iterations = golden_section_search(
            lambda x: x, 0.0, 1.0, x_tolerance=1.0e-10
        )
        self.assertLess(x_min, 1.0e-9)
        self.assertGreater(iterations, 0)

    def test_min_at_right_endpoint(self):
        # f increasing on [0, 1], minimum at the right edge (value 0).
        # The search converges toward 1 but always keeps interior points,
        # so x_min approaches 1 without ever equaling it exactly.
        x_min, f_min, iterations = golden_section_search(
            lambda x: -x + 1.0, 0.0, 1.0, x_tolerance=1.0e-10
        )
        self.assertGreater(x_min, 1.0 - 1.0e-9)
        self.assertLess(iterations, 100)

    def test_tiny_interval(self):
        # A sub-microsecond bracket around the minimum. Should still locate
        # the minimum to within float precision.
        center = 2.5
        x_min, f_min, _ = golden_section_search(
            lambda x: (x - center) ** 2,
            center - 1e-12,
            center + 1e-12,
            x_tolerance=1e-18,
        )
        self.assertAlmostEqual(x_min, center, places=10)

    def test_flat_function_picks_left_interior_point(self):
        # f(x) = 0 everywhere. By the implementation's tie-break rule
        # (f_left <= f_right picks the left branch), the loop drives the
        # bracket toward the left. The reported x_min is the better of the
        # two final interior points; for a flat function that is the left
        # one. We assert it lies within the bracket and equals 0 when
        # evaluated.
        x_min, f_min, iterations = golden_section_search(
            lambda x: 0.0, 0.0, 1.0, x_tolerance=1.0e-10
        )
        self.assertGreaterEqual(x_min, 0.0)
        self.assertLessEqual(x_min, 1.0)
        self.assertAlmostEqual(f_min, 0.0)
        self.assertGreater(iterations, 0)


class TestGoldenSectionSearchInputValidation(unittest.TestCase):
    """Bad inputs raise immediately, before any function evaluation."""

    def test_rejects_lower_not_less_than_upper(self):
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x ** 2, 1.0, 1.0)
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x ** 2, 2.0, 1.0)

    def test_rejects_non_positive_tolerance(self):
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x ** 2, 0.0, 1.0, x_tolerance=0.0)
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x ** 2, 0.0, 1.0, x_tolerance=-1.0)

    def test_rejects_non_positive_max_iterations(self):
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x ** 2, 0.0, 1.0, max_iterations=0)
        with self.assertRaises(ValueError):
            golden_section_search(lambda x: x ** 2, 0.0, 1.0, max_iterations=-3)

    def test_rejects_non_real_lower(self):
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, "0", 1.0)
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, None, 1.0)

    def test_rejects_non_real_upper(self):
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, 0.0, "1")
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, 0.0, None)

    def test_rejects_non_real_tolerance(self):
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, 0.0, 1.0, x_tolerance="small")
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, 0.0, 1.0, x_tolerance=None)

    def test_rejects_non_int_max_iterations(self):
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, 0.0, 1.0, max_iterations=50.0)
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, 0.0, 1.0, max_iterations="50")

    def test_rejects_bool_lower(self):
        # bool is a subclass of int; the library rejects it as a likely bug.
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, False, 1.0)

    def test_rejects_bool_upper(self):
        with self.assertRaises(TypeError):
            golden_section_search(lambda x: x ** 2, 0.0, True)

    def test_rejects_bool_tolerance(self):
        with self.assertRaises(TypeError):
            golden_section_search(
                lambda x: x ** 2, 0.0, 1.0, x_tolerance=False
            )

    def test_rejects_bool_max_iterations(self):
        with self.assertRaises(TypeError):
            golden_section_search(
                lambda x: x ** 2, 0.0, 1.0, max_iterations=True
            )


if __name__ == "__main__":
    unittest.main()
