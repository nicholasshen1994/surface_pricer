"""Pure Python implementation of the EDS smile slice.

The existing repository implementation uses the same parameterization but
normally delegates the slice to a SWIG/C++ wrapper. This module intentionally
keeps the numerical slice local to ``surface_pricer`` so the vanilla path can
run without the repository runtime.
"""

from dataclasses import dataclass
import math
from typing import Iterable, List

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq
from scipy.special import ndtr

from ..core.math.black import black_price
from ..core.math.implied_vol import implied_vol
from ..core.math.jaeckel import implied_vol_jaeckel


GL_X, GL_W = np.polynomial.legendre.leggauss(16)
PRECISION = 1.0e-8
MAX_ITERATIONS = 200
MINIMUM_VOL = 1.0e-5
ONE_MINUTE_VOL = 2.7557319223985893e-6
INVSQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _gl_integral(function, start: float, end: float, vector_function=None) -> float:
    if start == end:
        return 0.0
    half = 0.5 * (end - start)
    center = 0.5 * (end + start)
    if vector_function is not None:
        values = np.asarray(vector_function(half * GL_X + center), dtype=float)
        return float(half * np.dot(GL_W, values))
    return float(
        half
        * sum(
            weight * function(half * node + center)
            for node, weight in zip(GL_X, GL_W)
        )
    )


def _gl_integrals(vector_function, edges) -> np.ndarray:
    """Batched :func:`_gl_integral` over the intervals ``edges[i] .. edges[i+1]``.

    The same 16-point Gauss-Legendre rule, the same per-interval dot product as
    :func:`_gl_integral` - only evaluated for every interval at once: all the
    quadrature points go through ``vector_function`` in a single call and the
    reduction is one matrix product.  This is the hottest line of the local-vol
    build (the slice grid is rebuilt on every calibration iteration, ~13 times per
    slice), where the per-interval Python call + spline evaluation dominated.
    """
    edges = np.asarray(edges, dtype=float)
    lower = edges[:-1]
    upper = edges[1:]
    half = 0.5 * (upper - lower)
    center = 0.5 * (upper + lower)
    points = center[:, None] + half[:, None] * GL_X[None, :]
    values = np.asarray(vector_function(points.ravel()), dtype=float).reshape(points.shape)
    return half * (values @ GL_W)


def _rk4(function, x0: float, y0: float, x1: float, steps: int = 100) -> float:
    if x1 == x0:
        return y0
    steps = max(1, int(steps))
    h = (x1 - x0) / steps
    x = x0
    y = y0
    for _ in range(steps):
        k1 = h * function(x, y)
        k2 = h * function(x + 0.5 * h, y + 0.5 * k1)
        k3 = h * function(x + 0.5 * h, y + 0.5 * k2)
        k4 = h * function(x + h, y + k3)
        y += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        x += h
    return y


def _norm_cdf_diff(x1: float, x2: float) -> float:
    if x1 * x2 < 0.0 or x1 < 0.0:
        return float(ndtr(x1) - ndtr(x2))
    return float(ndtr(-x2) - ndtr(-x1))


def _smooth_max(a: float, b: float, step: float) -> float:
    if a >= b + step:
        return a
    if a <= b:
        return b
    x = (a - b) / step * 2.0 - 1.0
    x2 = x * x
    x4 = x2 * x2
    x6 = x2 * x4
    sx = 0.03125 * x6 - 0.15625 * x4 + 0.46875 * x2 + 0.5 * x + 0.15625
    return sx * (a - b) + b


def _quadratic_intervals(a: float, b: float, c: float, start: float, end: float):
    """Split an interval into negative and positive quadratic regions."""
    reverse = start > end
    left, right = min(start, end), max(start, end)

    if abs(a) < 1.0e-14:
        if abs(b) < 1.0e-14:
            if c > 0.0:
                return [], [[start, end]], None
            else:
                return [[start, end]], [], None
        else:
            root = -c / b
            if b > 0.0:
                if root <= left:
                    negative, positive, roots = [], [[left, right]], None
                elif root >= right:
                    negative, positive, roots = [[left, right]], [], None
                else:
                    negative, positive, roots = [[left, root]], [[root, right]], [root]
            else:
                result = _quadratic_intervals(0.0, -b, -c, start, end)
                return result[1], result[0], result[2]
    elif a > 0.0:
        discriminant = b * b - 4.0 * a * c
        if discriminant > 0.0:
            root_1 = (-b - math.sqrt(discriminant)) / (2.0 * a)
            root_2 = (-b + math.sqrt(discriminant)) / (2.0 * a)
            if root_2 <= left or root_1 >= right:
                negative, positive, roots = [], [[left, right]], None
            else:
                neg_start = max(left, root_1)
                neg_end = min(right, root_2)
                negative = [[neg_start, neg_end]]
                if neg_start > left and neg_end < right:
                    positive, roots = [[left, neg_start], [neg_end, right]], [root_1, root_2]
                elif neg_start > left:
                    positive, roots = [[left, neg_start]], [root_1]
                elif neg_end < right:
                    positive, roots = [[neg_end, right]], [root_2]
                else:
                    positive, roots = [], None
        else:
            negative, positive, roots = [], [[left, right]], None
    else:
        result = _quadratic_intervals(-a, -b, -c, start, end)
        return result[1], result[0], result[2]
    if reverse:
        # ``np.flip`` in the repository implementation reverses both the
        # interval order and each interval's endpoints.  Preserving that
        # orientation is essential for left-wing integrals.
        return (
            np.flip(np.asarray(negative, dtype=float)).tolist(),
            np.flip(np.asarray(positive, dtype=float)).tolist(),
            roots,
        )
    return negative, positive, roots


def _find_pchip_derivatives(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = x[1:] - x[:-1]
    slopes = (y[1:] - y[:-1]) / h
    signs = np.sign(slopes)
    condition = (signs[1:] != signs[:-1]) | (slopes[1:] == 0) | (slopes[:-1] == 0)
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 / slopes[:-1] + w2 / slopes[1:]) / (w1 + w2)

    derivative = np.zeros_like(y)
    derivative[1:-1][~condition] = 1.0 / harmonic[~condition]

    def edge(h0, h1, m0, m1):
        value = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        if np.sign(value) != np.sign(m0):
            return 0.0
        if np.sign(m0) != np.sign(m1) and abs(value) > 3.0 * abs(m0):
            return 3.0 * m0
        return value

    derivative[0] = edge(h[0], h[1], slopes[0], slopes[1])
    derivative[-1] = edge(h[-1], h[-2], slopes[-1], slopes[-2])
    return derivative


class _MonotonicSpline(CubicHermiteSpline):
    """PCHIP-like spline with linear extrapolation."""

    def __init__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        x_extended = np.concatenate(([2.0 * x[0] - x[1]], x, [2.0 * x[-1] - x[-2]]))
        y_extended = np.concatenate(([2.0 * y[0] - y[1]], y, [2.0 * y[-1] - y[-2]]))
        super().__init__(x_extended, y_extended, _find_pchip_derivatives(x_extended, y_extended))


@dataclass
class EDSSliceParameters:
    ref_strike: float
    forward: float
    vol_atmf: float
    tau: float
    skew: float = 0.0
    conv: float = 0.0
    left_skew_1: float = 0.0
    left_skew_2: float = 0.0
    right_skew_1: float = 0.0
    right_skew_2: float = 0.0


class EDSSabrSlice:
    """One calibrated EDS SABR smile slice."""

    def __init__(
        self,
        ref_strike: float,
        forward: float,
        vol_atmf: float,
        tau: float = 1.0,
        skew: float = 0.0,
        conv: float = 0.0,
        left_skew_1: float = 0.0,
        left_skew_2: float = 0.0,
        right_skew_1: float = 0.0,
        right_skew_2: float = 0.0,
        calibrate: bool = True,
    ):
        self.ref_strike = float(forward if ref_strike == 0.0 else ref_strike)
        self.forward = float(forward)
        self.vol_atmf = max(float(vol_atmf), MINIMUM_VOL)
        self.tau = max(float(tau), ONE_MINUTE_VOL)
        self.sqrttau = math.sqrt(self.tau)
        self.skew = float(skew)
        self.conv = float(conv)
        self.left_skew_1 = float(left_skew_1)
        self.left_skew_2 = float(left_skew_2)
        self.right_skew_1 = float(right_skew_1)
        self.right_skew_2 = float(right_skew_2)

        self.left_cut_1 = 1.0
        self.left_cut_2 = 4.0
        self.left_cut_3 = 10.0
        self.left_bound_3_min = 0.15
        self.left_nsmooth = 0.05
        self.right_cut_1 = 1.0
        self.right_cut_2 = 4.0
        self.right_cut_3 = 10.0
        self.right_bound_3_min = 0.15
        self.right_nsmooth = 0.05

        self.mu = -0.5 * self.vol_atmf ** 2 * self.tau
        self.stdev = self.vol_atmf * self.sqrttau
        self._is_zero_stdev = abs(self.stdev) < 1.0e-14
        self._epsilon = 0.05 * self.stdev
        self.left_bound_1 = -self.left_cut_1 * self.stdev
        self.left_skew_dr = self.skew - self.conv - 4.0 / 3.0 * self.left_skew_1
        self.left_bound_2 = -self.left_cut_2 * self.stdev
        self.left_bound_3 = -_smooth_max(
            self.left_cut_3 * self.stdev,
            self.left_bound_3_min,
            self.left_nsmooth,
        )
        self.right_bound_1 = self.right_cut_1 * self.stdev
        self.right_skew_dr = self.skew + self.conv + 4.0 / 3.0 * self.right_skew_1
        self.right_bound_2 = self.right_cut_2 * self.stdev
        self.right_bound_3 = _smooth_max(
            self.right_cut_3 * self.stdev,
            self.right_bound_3_min,
            self.right_nsmooth,
        )
        self.ref_strike_div_fwd = self.ref_strike / self.forward
        self.ln_ref_strike_div_fwd = math.log(self.ref_strike_div_fwd)

        self._int_vol_add = np.array(
            [
                -4.0 * self.stdev * (self.skew - self.conv - self.left_skew_1)
                + self.left_skew_2 * (self.left_bound_2 - self.left_bound_3),
                -4.0 * self.stdev * (self.skew - self.conv - self.left_skew_1),
                -self.stdev * (self.skew - self.conv),
                0.0,
                self.stdev * (self.skew + self.conv),
                4.0 * self.stdev * (self.skew + self.conv + self.right_skew_1),
                4.0 * self.stdev * (self.skew + self.conv + self.right_skew_1)
                + self.right_skew_2 * (self.right_bound_3 - self.right_bound_2),
            ],
            dtype=float,
        )
        self.int_vol_ref = self.vol_atmf
        self.int_vol_knot_array = self._int_vol_add + self.int_vol_ref
        self.key_y_grid = None
        self.key_x_grid = None
        self.sum_prob = None
        self.exp_y_adj = None
        self.key_y_integral = None
        self.exp_y_grid = None
        self.interpolator = None
        self.y_int = None
        self.key_x_prob = None
        self.anchor_int = {}
        self.split_y_intervals = {}
        self.key_y_int_from_left = None
        self.key_y_int_to_right = None
        self.key_x_prob_from_left = None
        self.key_x_prob_to_right = None
        self.y_prob_add_stt = None
        self.y_prob_add_end = None

        self.smileless = all(
            abs(value) < 1.0e-14
            for value in (
                self.skew,
                self.conv,
                self.left_skew_1,
                self.left_skew_2,
                self.right_skew_1,
                self.right_skew_2,
            )
        )
        if not self.smileless:
            if calibrate:
                self._calibrate()
            else:
                self._generate_xy_grid()

    def _calibrate(self):
        self._generate_xy_grid()
        current = self._vol_at_moneyness(self.ref_strike_div_fwd)
        count = 0
        solution = self.vol_atmf
        while np.isfinite(current) and abs(current - self.vol_atmf) > PRECISION and count < 20:
            solution *= self.vol_atmf / current
            if solution < 0.001 or solution > 10.0:
                break
            self._set_int_vol(solution)
            current = self._vol_at_moneyness(self.ref_strike_div_fwd)
            count += 1

        if not np.isfinite(current) or abs(current - self.vol_atmf) > 0.01:
            def objective(value):
                self._set_int_vol(value)
                return self._vol_at_moneyness(self.ref_strike_div_fwd) - self.vol_atmf

            try:
                solution = brentq(objective, 0.001, 10.0, xtol=PRECISION, maxiter=200)
            except Exception:
                solution = self.vol_atmf

        # The grid is rebuilt only when the vol level actually moved: after the
        # fixed-point loop it already matches ``solution`` (the loop's last act was
        # to rebuild for it), so the unconditional rebuild this used to do was one
        # whole grid construction - the single most expensive step here - per slice
        # for nothing.  A ``break``/``brentq`` leaves it on the last *probe*, and
        # then it is rebuilt.
        if self.int_vol_ref != solution:
            self._set_int_vol(solution)

    def _set_int_vol(self, value: float) -> None:
        """Point the slice at ``int_vol_ref = value`` and rebuild its grid."""
        self.int_vol_ref = value
        self.int_vol_knot_array = self._int_vol_add + self.int_vol_ref
        self._generate_xy_grid()

    def _local_vol(self, value: float) -> float:
        if value <= self.left_bound_3:
            output = self.int_vol_knot_array[0]
        elif value >= self.right_bound_3:
            output = self.int_vol_knot_array[-1]
        elif value <= self.left_bound_2:
            output = self.int_vol_knot_array[1] + self.left_skew_2 * (self.left_bound_2 - value)
        elif value >= self.right_bound_2:
            output = self.int_vol_knot_array[-2] + self.right_skew_2 * (value - self.right_bound_2)
        elif value <= self.left_bound_1:
            output = self.int_vol_knot_array[2] + self.left_skew_dr * (value - self.left_bound_1)
        elif value >= self.right_bound_1:
            output = self.int_vol_knot_array[-3] + self.right_skew_dr * (value - self.right_bound_1)
        elif self._is_zero_stdev:
            output = self.int_vol_knot_array[3]
        else:
            output = (
                self.int_vol_knot_array[3]
                + self.skew * value
                + self.conv * value * value / self.stdev
            )
        if output >= self._epsilon:
            return output
        return (output - 2.0 * self._epsilon) / (2.0 * output - 3.0 * self._epsilon) * self._epsilon

    def _prepare_anchor_integrals(self):
        self.anchor_int = {}
        self.split_y_intervals = {}
        initial = [
            -math.inf,
            self.left_bound_3,
            self.left_bound_2,
            self.left_bound_1,
            0.0,
            self.right_bound_1,
            self.right_bound_2,
            self.right_bound_3,
            math.inf,
        ]
        anchors = set(initial)
        parameter_sets = [
            (0.0, 0.0, self.int_vol_knot_array[0]),
            (
                0.0,
                -self.left_skew_2,
                self.int_vol_knot_array[1] + self.left_skew_2 * self.left_bound_2,
            ),
            (
                0.0,
                self.left_skew_dr,
                self.int_vol_knot_array[2] - self.left_skew_dr * self.left_bound_1,
            ),
            (
                (0.0 if self._is_zero_stdev else self.conv / self.stdev),
                (0.0 if self._is_zero_stdev else self.skew),
                self.int_vol_knot_array[3],
            ),
            (
                0.0,
                self.right_skew_dr,
                self.int_vol_knot_array[-3] - self.right_skew_dr * self.right_bound_1,
            ),
            (
                0.0,
                self.right_skew_2,
                self.int_vol_knot_array[-2] - self.right_skew_2 * self.right_bound_2,
            ),
            (0.0, 0.0, self.int_vol_knot_array[-1]),
        ]
        for index in range(len(initial) // 2):
            right_interval = (initial[4 + index], initial[5 + index])
            left_interval = (initial[4 - index], initial[3 - index])
            right_parameters = parameter_sets[3 + index]
            left_parameters = parameter_sets[3 - index]
            right_regions = _quadratic_intervals(
                right_parameters[0],
                right_parameters[1],
                right_parameters[2] - self._epsilon,
                *right_interval,
            )
            left_regions = _quadratic_intervals(
                left_parameters[0],
                left_parameters[1],
                left_parameters[2] - self._epsilon,
                *left_interval,
            )
            self.split_y_intervals[right_interval] = [right_regions[0], right_regions[1]]
            self.split_y_intervals[left_interval] = [left_regions[0], left_regions[1]]
            if right_regions[2]:
                anchors.update(right_regions[2])
            if left_regions[2]:
                anchors.update(left_regions[2])
        values = self._integral_y_to_x(list(anchors), add_constant=False)
        self.anchor_int = {anchor: values[index] for index, anchor in enumerate(anchors)}

    def _integral_y_to_x(self, values: Iterable[float], add_constant: bool = True):
        def reciprocal_quadratic(a, b, c, start, end):
            if abs(a) < 1.0e-14:
                if abs(b) < 1.0e-14:
                    return (end - start) / c
                return math.log((b * end + c) / (b * start + c)) / b
            discriminant = b * b - 4.0 * a * c
            if abs(discriminant) < 1.0e-14:
                root = -b / (2.0 * a)
                return -1.0 / a / (end - root) + 1.0 / a / (start - root)
            if discriminant > 0.0:
                root_1 = (-b + math.sqrt(discriminant)) / (2.0 * a)
                root_2 = (-b - math.sqrt(discriminant)) / (2.0 * a)
                return math.log(
                    (end - root_1) * (start - root_2)
                    / ((end - root_2) * (start - root_1))
                ) / a / (root_1 - root_2)
            real = -b / (2.0 * a)
            imaginary = math.sqrt(-discriminant) / (2.0 * a)
            return (
                math.atan((end - real) / imaginary)
                - math.atan((start - real) / imaginary)
            ) / a / imaginary

        def ordinary(a, b, c, start, end):
            return reciprocal_quadratic(a, b, c, start, end) / self.sqrttau

        def regularized(a, b, c, start, end):
            adjusted_c = c - 2.0 * self._epsilon
            return (
                reciprocal_quadratic(a, b, adjusted_c, start, end)
                + 2.0 * (end - start) / self._epsilon
            ) / self.sqrttau

        result = []
        adjustment = -self.ln_ref_strike_div_fwd + self.mu
        for y_adj in values:
            value = 0.0
            if y_adj != adjustment and add_constant:
                if adjustment in self.anchor_int:
                    value = -self.anchor_int[adjustment]
                else:
                    value = -self._integral_y_to_x([adjustment], add_constant=False)[0]
                    self.anchor_int[adjustment] = -value
            if y_adj in self.anchor_int:
                result.append(
                    self.anchor_int[y_adj] - self.anchor_int[adjustment]
                    if add_constant
                    else self.anchor_int[y_adj]
                )
                continue
            if np.isinf(y_adj):
                result.append(float(y_adj))
                continue

            if self.left_bound_1 <= y_adj <= self.right_bound_1:
                interval = (0.0, y_adj)
                known = None
                parameters = (
                    (0.0, 0.0, self.int_vol_knot_array[3])
                    if self._is_zero_stdev
                    else (self.conv / self.stdev, self.skew, self.int_vol_knot_array[3])
                )
            elif self.right_bound_1 < y_adj <= self.right_bound_2:
                known = self.right_bound_1
                interval = (known, y_adj)
                parameters = (
                    0.0,
                    self.right_skew_dr,
                    self.int_vol_knot_array[-3] - self.right_skew_dr * self.right_bound_1,
                )
            elif self.right_bound_2 < y_adj <= self.right_bound_3:
                known = self.right_bound_2
                interval = (known, y_adj)
                parameters = (
                    0.0,
                    self.right_skew_2,
                    self.int_vol_knot_array[-2] - self.right_skew_2 * self.right_bound_2,
                )
            elif y_adj > self.right_bound_3:
                known = self.right_bound_3
                interval = (known, y_adj)
                parameters = (0.0, 0.0, self.int_vol_knot_array[-1])
            elif self.left_bound_2 <= y_adj < self.left_bound_1:
                known = self.left_bound_1
                interval = (known, y_adj)
                parameters = (
                    0.0,
                    self.left_skew_dr,
                    self.int_vol_knot_array[2] - self.left_skew_dr * self.left_bound_1,
                )
            elif self.left_bound_3 <= y_adj < self.left_bound_2:
                known = self.left_bound_2
                interval = (known, y_adj)
                parameters = (
                    0.0,
                    -self.left_skew_2,
                    self.int_vol_knot_array[1] + self.left_skew_2 * self.left_bound_2,
                )
            else:
                known = self.left_bound_3
                interval = (known, y_adj)
                parameters = (0.0, 0.0, self.int_vol_knot_array[0])

            if known is not None:
                value += self._integral_y_to_x([known], add_constant=False)[0]
            regular_regions, ordinary_regions, _ = _quadratic_intervals(
                parameters[0],
                parameters[1],
                parameters[2] - self._epsilon,
                *interval,
            )
            for region in ordinary_regions:
                value += ordinary(*parameters, *region)
            for region in regular_regions:
                value += regularized(*parameters, *region)
            result.append(value)
        return np.asarray(result, dtype=float)

    def _integral_x_to_y(self, values: Iterable[float]):
        def inverse_quadratic_integral(a, b, c, target, lower):
            if abs(a) < 1.0e-14:
                if abs(b) < 1.0e-14:
                    return c * target + lower
                return (math.exp(b * target) * (b * lower + c) - c) / b
            discriminant = b * b - 4.0 * a * c
            if abs(discriminant) < 1.0e-14:
                root = -b / (2.0 * a)
                return (lower - root) / (1.0 - a * target * (lower - root)) + root
            if discriminant > 0.0:
                root_1 = (-b + math.sqrt(discriminant)) / (2.0 * a)
                root_2 = (-b - math.sqrt(discriminant)) / (2.0 * a)
                return (
                    root_2
                    - root_1
                ) / (
                    math.exp(target * a * (root_1 - root_2))
                    * (lower - root_1) / (lower - root_2)
                    - 1.0
                ) + root_2
            real = -b / (2.0 * a)
            imaginary = math.sqrt(-discriminant) / (2.0 * a)
            return math.tan(
                target * a * imaginary
                + math.atan((lower - real) / imaginary)
            ) * imaginary + real

        def get_interval(x):
            boundaries = [
                self.left_bound_1,
                self.right_bound_1,
                self.right_bound_2,
                self.right_bound_3,
                self.left_bound_2,
                self.left_bound_3,
            ]
            x_boundaries = self._integral_y_to_x(boundaries)
            if x_boundaries[0] <= x <= x_boundaries[1]:
                y_interval = (
                    0.0,
                    self.left_bound_1
                    if x <= self._integral_y_to_x([0.0])[0]
                    else self.right_bound_1,
                )
                parameters = (
                    (0.0, 0.0, self.int_vol_knot_array[3])
                    if self._is_zero_stdev
                    else (self.conv / self.stdev, self.skew, self.int_vol_knot_array[3])
                )
            elif x_boundaries[1] < x <= x_boundaries[2]:
                y_interval = (self.right_bound_1, self.right_bound_2)
                parameters = (0.0, self.right_skew_dr, self.int_vol_knot_array[-3] - self.right_skew_dr * self.right_bound_1)
            elif x_boundaries[2] < x <= x_boundaries[3]:
                y_interval = (self.right_bound_2, self.right_bound_3)
                parameters = (0.0, self.right_skew_2, self.int_vol_knot_array[-2] - self.right_skew_2 * self.right_bound_2)
            elif x > x_boundaries[3]:
                y_interval = (self.right_bound_3, math.inf)
                parameters = (0.0, 0.0, self.int_vol_knot_array[-1])
            elif x_boundaries[4] <= x < x_boundaries[0]:
                y_interval = (self.left_bound_1, self.left_bound_2)
                parameters = (0.0, self.left_skew_dr, self.int_vol_knot_array[2] - self.left_skew_dr * self.left_bound_1)
            elif x_boundaries[5] <= x < x_boundaries[4]:
                y_interval = (self.left_bound_2, self.left_bound_3)
                parameters = (0.0, -self.left_skew_2, self.int_vol_knot_array[1] + self.left_skew_2 * self.left_bound_2)
            else:
                y_interval = (self.left_bound_3, -math.inf)
                parameters = (0.0, 0.0, self.int_vol_knot_array[0])
            regular, ordinary, _ = _quadratic_intervals(
                parameters[0],
                parameters[1],
                parameters[2] - self._epsilon,
                *y_interval,
            )
            if not regular:
                return ordinary[0][0], False, parameters
            if not ordinary:
                return regular[0][0], True, parameters

            # A local-vol branch can cross the regularization threshold
            # inside a piecewise interval.  The inverse integral must start
            # from the sub-interval containing x, not from the first one.
            for interval in regular:
                x_range = self._integral_y_to_x(interval)
                if min(x_range) <= x <= max(x_range):
                    return interval[0], True, parameters
            for interval in ordinary:
                x_range = self._integral_y_to_x(interval)
                if min(x_range) <= x <= max(x_range):
                    return interval[0], False, parameters
            raise ValueError("normal-driver value falls outside EDS local-vol intervals")

        def function(_, y):
            return self.sqrttau * self._local_vol(y - self.ln_ref_strike_div_fwd + self.mu)

        adjustment = -self.ln_ref_strike_div_fwd + self.mu
        result = []
        x_start, y_start = 0.0, 0.0
        for x in values:
            lower, is_regular, parameters = get_interval(x)
            if is_regular:
                y = _rk4(function, y_start, x_start, x, steps=100)
            else:
                boundary_x = self._integral_y_to_x([lower], add_constant=True)[0]
                target = (x - boundary_x) * self.sqrttau
                y = inverse_quadratic_integral(*parameters, target, lower) - adjustment
            x_start, y_start = x, y
            result.append(y)
        return np.asarray(result, dtype=float)

    def _generate_xy_grid(self):
        self._prepare_anchor_integrals()
        forward_keys = self._integral_x_to_y([2.0, 4.0, 7.0, 10.0])
        backward_keys = self._integral_x_to_y([-2.0, -4.0, -7.0, -10.0])
        self.key_y_grid = np.concatenate(
            [
                np.linspace(backward_keys[3], max(self.left_bound_3, backward_keys[2]), 21)[:-1],
                np.linspace(max(self.left_bound_3, backward_keys[2]), max(self.left_bound_2, backward_keys[1]), 21)[:-1],
                np.linspace(max(self.left_bound_2, backward_keys[1]), max(self.left_bound_1, backward_keys[0]), 21)[:-1],
                np.linspace(max(self.left_bound_1, backward_keys[0]), 0.0, 21),
                np.linspace(0.0, min(self.right_bound_1, forward_keys[0]), 21)[1:],
                np.linspace(min(self.right_bound_1, forward_keys[0]), min(self.right_bound_2, forward_keys[1]), 21)[1:],
                np.linspace(min(self.right_bound_2, forward_keys[1]), min(self.right_bound_3, forward_keys[2]), 21)[1:],
                np.linspace(min(self.right_bound_3, forward_keys[2]), forward_keys[3], 21)[1:],
            ]
        )
        adjusted = self.key_y_grid - self.ln_ref_strike_div_fwd + self.mu
        self.key_x_grid = self._integral_y_to_x(adjusted, add_constant=True)
        self.key_x_grid = np.asarray(self.key_x_grid, dtype=float)
        for index in range(1, len(self.key_x_grid)):
            if self.key_x_grid[index] <= self.key_x_grid[index - 1]:
                self.key_x_grid[index] = self.key_x_grid[index - 1] + 1.0e-12

        self.interpolator = _MonotonicSpline(self.key_x_grid, self.key_y_grid)
        self.y_int = lambda x: math.exp(float(self.interpolator(x)) - 0.5 * x * x) * INVSQRT2PI
        self.y_int_vec = (
            lambda x: np.exp(
                np.asarray(self.interpolator(x), dtype=float)
                - 0.5 * np.asarray(x, dtype=float) ** 2
            )
            * INVSQRT2PI
        )
        self.key_y_integral = _gl_integrals(self.y_int_vec, self.key_x_grid)
        upper_x = self.key_x_grid[1:]
        lower_x = self.key_x_grid[:-1]
        self.key_x_prob = np.where(
            (upper_x * lower_x < 0.0) | (upper_x < 0.0),
            ndtr(upper_x) - ndtr(lower_x),
            ndtr(-lower_x) - ndtr(-upper_x),
        )
        self.key_x_prob[0] = ndtr(self.key_x_grid[1])
        self.key_x_prob[-1] = ndtr(-self.key_x_grid[-2])
        slope_start = (self.key_y_grid[1] - self.key_y_grid[0]) / (self.key_x_grid[1] - self.key_x_grid[0])
        intercept_start = self.key_y_grid[0] - slope_start * self.key_x_grid[0]
        slope_end = (self.key_y_grid[-1] - self.key_y_grid[-2]) / (self.key_x_grid[-1] - self.key_x_grid[-2])
        intercept_end = self.key_y_grid[-1] - slope_end * self.key_x_grid[-1]
        self.y_prob_add_stt = ndtr(self.key_x_grid[0] - slope_start) * math.exp(
            0.5 * slope_start * slope_start + intercept_start
        )
        self.y_prob_add_end = ndtr(slope_end - self.key_x_grid[-1]) * math.exp(
            0.5 * slope_end * slope_end + intercept_end
        )
        self.key_y_integral[0] += self.y_prob_add_stt
        self.key_y_integral[-1] += self.y_prob_add_end
        self.sum_prob = float(self.key_x_prob.sum())
        self.exp_y_adj = float(self.key_y_integral.sum() / self.sum_prob)
        self.key_y_integral /= self.exp_y_adj
        self.exp_y_grid = np.exp(self.key_y_grid) / self.exp_y_adj
        self.key_y_int_from_left = self.key_y_integral.cumsum()
        # Suffix sums in O(n): the O(n^2) Python loop this replaces (one numpy
        # ``sum`` per index) cost more than everything else in the grid build put
        # together after the quadrature itself.  Same values to rounding.
        self.key_y_int_to_right = np.cumsum(self.key_y_integral[::-1])[::-1]
        self.key_x_prob_from_left = self.key_x_prob.cumsum()
        self.key_x_prob_to_right = np.cumsum(self.key_x_prob[::-1])[::-1]

    def _vol_at_moneyness(self, moneyness: float) -> float:
        if moneyness < self.exp_y_grid[1]:
            return self._vol_at_moneyness(self.exp_y_grid[1])
        if moneyness > self.exp_y_grid[-2]:
            return self._vol_at_moneyness(self.exp_y_grid[-2])

        grid_index = int(np.searchsorted(self.exp_y_grid, moneyness, side="left"))
        x_base = self.key_x_grid[grid_index - 1]
        x_top = self.key_x_grid[grid_index]
        if abs(self.exp_y_grid[grid_index] - moneyness) < PRECISION:
            x_value = self.key_x_grid[grid_index]
        else:
            target = math.log(moneyness * self.exp_y_adj)
            x_value = brentq(
                lambda x: float(self.interpolator(x)) - target,
                x_base,
                x_top,
                xtol=PRECISION,
                rtol=PRECISION,
                maxiter=MAX_ITERATIONS,
            )

        if moneyness <= 1.0:
            if grid_index <= 1:
                sum_y = (
                    _gl_integral(
                        self.y_int,
                        self.key_x_grid[0],
                        x_value,
                        vector_function=self.y_int_vec,
                    )
                    + self.y_prob_add_stt
                ) / self.exp_y_adj
                sum_x = moneyness * ndtr(x_value)
            else:
                sum_y = self.key_y_integral[: grid_index - 1].sum() + _gl_integral(
                    self.y_int,
                    x_base,
                    x_value,
                    vector_function=self.y_int_vec,
                ) / self.exp_y_adj
                sum_x = moneyness * (
                    self.key_x_prob[: grid_index - 1].sum()
                    + _norm_cdf_diff(x_value, x_base)
                )
            put_price = (sum_x - sum_y) / self.sum_prob
            return self._invert(put_price, moneyness, "put")

        if grid_index >= len(self.exp_y_grid) - 2:
            sum_y = (
                _gl_integral(
                    self.y_int,
                    x_value,
                    self.key_x_grid[-1],
                    vector_function=self.y_int_vec,
                )
                + self.y_prob_add_end
            ) / self.exp_y_adj
            sum_x = moneyness * ndtr(-x_value)
        else:
            sum_y = self.key_y_integral[grid_index:].sum() + _gl_integral(
                self.y_int,
                x_value,
                x_top,
                vector_function=self.y_int_vec,
            ) / self.exp_y_adj
            sum_x = moneyness * (
                self.key_x_prob[grid_index:].sum()
                + _norm_cdf_diff(x_top, x_value)
            )
        call_price = (sum_y - sum_x) / self.sum_prob
        return self._invert(call_price, moneyness, "call")

    def _invert(self, price: float, moneyness: float, option_type: str) -> float:
        """Black IV of an undiscounted unit-forward price.

        Jaeckel's analytic inversion first: it is a closed form (~1us) where the
        bisection needs ~60 Black evaluations (~1.5ms), and the local-vol table
        inverts one vol per strike per slice.  The bisection stays as the
        fallback for prices outside the no-arbitrage band, where the analytic
        algorithm reports ``None`` instead of a vol.
        """
        fast = implied_vol_jaeckel(price, 1.0, moneyness, self.tau, option_type)
        if fast is not None:
            return fast
        return implied_vol(price, 1.0, moneyness, self.tau, 1.0, option_type)

    def get_implied_vol(self, strikes: Iterable[float]) -> np.ndarray:
        strike_array = np.atleast_1d(np.asarray(strikes, dtype=float))
        if self.smileless:
            result = np.full(strike_array.shape, self.vol_atmf, dtype=float)
        else:
            result = np.asarray(
                [self._vol_at_moneyness(strike / self.forward) for strike in strike_array],
                dtype=float,
            )
        if np.any(~np.isfinite(result)):
            raise ValueError("EDS implied volatility contains NaN or Inf")
        return result
