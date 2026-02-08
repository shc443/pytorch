#!/usr/bin/env python3
"""
Tests for modified Bessel functions with arbitrary order.
torch.special.modified_bessel_i(x, nu) and torch.special.modified_bessel_k(x, nu)

Issue #76324: Bessel and Related Functions
"""

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
)
import numpy as np

# Only import scipy if available
try:
    from scipy import special as scipy_special
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestModifiedBesselFunctions(TestCase):
    """Tests for modified Bessel functions I_nu and K_nu with arbitrary order."""

    def _skip_if_no_scipy(self):
        if not HAS_SCIPY:
            self.skipTest("scipy not available")

    # =========================================================================
    # I_nu tests
    # =========================================================================

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_integer_orders(self, device, dtype):
        """Test I_nu for integer orders 0-10."""
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        for nu_val in range(11):
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_i(x, nu)
            expected = torch.tensor(
                [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype
            )
            self.assertEqual(result, expected, rtol=1e-5, atol=1e-8)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_half_integer_orders(self, device, dtype):
        """Test I_nu for half-integer orders."""
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        for nu_val in [0.5, 1.5, 2.5, 3.5]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_i(x, nu)
            expected = torch.tensor(
                [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype
            )
            self.assertEqual(result, expected, rtol=1e-5, atol=1e-8)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_arbitrary_orders(self, device, dtype):
        """Test I_nu for arbitrary orders like 2.73, 12.73."""
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 15, 50, device=device, dtype=dtype)
        for nu_val in [2.73, 5.17, 12.73]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_i(x, nu)
            expected = torch.tensor(
                [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype
            )
            self.assertEqual(result, expected, rtol=1e-5, atol=1e-8)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_edge_cases(self, device, dtype):
        """Test I_nu edge cases: x=0, negative nu."""
        # I_0(0) = 1
        x = torch.tensor([0.0], device=device, dtype=dtype)
        nu = torch.tensor([0.0], device=device, dtype=dtype)
        result = torch.special.modified_bessel_i(x, nu)
        self.assertEqual(result, torch.tensor([1.0], device=device, dtype=dtype))

        # I_nu(0) = 0 for nu != 0
        nu = torch.tensor([1.0], device=device, dtype=dtype)
        result = torch.special.modified_bessel_i(x, nu)
        self.assertEqual(result, torch.tensor([0.0], device=device, dtype=dtype))

    # =========================================================================
    # K_nu tests
    # =========================================================================

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_integer_orders(self, device, dtype):
        """Test K_nu for integer orders 0-10."""
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        for nu_val in range(11):
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_k(x, nu)
            expected = torch.tensor(
                [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype
            )
            self.assertEqual(result, expected, rtol=1e-5, atol=1e-8)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_half_integer_orders(self, device, dtype):
        """Test K_nu for half-integer orders."""
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        for nu_val in [0.5, 1.5, 2.5, 3.5]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_k(x, nu)
            expected = torch.tensor(
                [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype
            )
            self.assertEqual(result, expected, rtol=1e-5, atol=1e-8)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_arbitrary_orders(self, device, dtype):
        """Test K_nu for arbitrary orders like 2.73, 12.73 (Matern kernel use case)."""
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 20, 100, device=device, dtype=dtype)
        for nu_val in [2.73, 5.17, 12.73]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_k(x, nu)
            expected = torch.tensor(
                [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype
            )
            self.assertEqual(result, expected, rtol=1e-5, atol=1e-8)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_edge_cases(self, device, dtype):
        """Test K_nu edge cases: x=0 -> inf."""
        x = torch.tensor([0.0], device=device, dtype=dtype)
        nu = torch.tensor([1.0], device=device, dtype=dtype)
        result = torch.special.modified_bessel_k(x, nu)
        self.assertTrue(torch.isinf(result).all())

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_symmetry(self, device, dtype):
        """Test K_{-nu}(x) = K_nu(x)."""
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        nu_pos = torch.full_like(x, 2.73)
        nu_neg = torch.full_like(x, -2.73)

        result_pos = torch.special.modified_bessel_k(x, nu_pos)
        result_neg = torch.special.modified_bessel_k(x, nu_neg)

        self.assertEqual(result_pos, result_neg, rtol=1e-10, atol=1e-10)

    # =========================================================================
    # Near-integer order tests (numerical stability at boundaries)
    # =========================================================================

    @dtypes(torch.float64)
    def test_modified_bessel_i_near_integer_orders(self, device, dtype):
        """Test I_nu continuity for orders near integers (e.g., 5.00001)."""
        self._skip_if_no_scipy()
        x = torch.tensor([1.0, 2.0, 5.0], device=device, dtype=dtype)

        # Test near-integer orders
        near_integer_cases = [
            (0.0, 1e-8),      # Near 0
            (0.0, 1e-5),
            (0.0, 0.0001),
            (1.0, 1e-8),      # Near 1
            (1.0, 1e-5),
            (1.0, 0.0001),
            (5.0, 1e-8),      # Near 5
            (5.0, 1e-5),
            (5.0, 0.0001),
        ]

        for base_nu, eps in near_integer_cases:
            for sign in [1, -1]:
                nu_val = base_nu + sign * eps
                nu = torch.full_like(x, nu_val)
                result = torch.special.modified_bessel_i(x, nu)
                expected = torch.tensor(
                    [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                    device=device,
                    dtype=dtype
                )
                self.assertEqual(
                    result, expected, rtol=1e-5, atol=1e-8,
                    msg=f"Failed for nu={nu_val}"
                )

    @dtypes(torch.float64)
    def test_modified_bessel_k_near_integer_orders(self, device, dtype):
        """Test K_nu continuity for orders near integers (e.g., 5.00001)."""
        self._skip_if_no_scipy()
        x = torch.tensor([1.0, 2.0, 5.0], device=device, dtype=dtype)

        # Test near-integer orders
        near_integer_cases = [
            (0.0, 1e-8),      # Near 0
            (0.0, 1e-5),
            (0.0, 0.0001),
            (1.0, 1e-8),      # Near 1
            (1.0, 1e-5),
            (1.0, 0.0001),
            (5.0, 1e-8),      # Near 5
            (5.0, 1e-5),
            (5.0, 0.0001),
        ]

        for base_nu, eps in near_integer_cases:
            for sign in [1, -1]:
                nu_val = base_nu + sign * eps
                nu = torch.full_like(x, nu_val)
                result = torch.special.modified_bessel_k(x, nu)
                expected = torch.tensor(
                    [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                    device=device,
                    dtype=dtype
                )
                self.assertEqual(
                    result, expected, rtol=1e-5, atol=1e-8,
                    msg=f"Failed for nu={nu_val}"
                )

    @dtypes(torch.float64)
    def test_modified_bessel_continuity_at_integers(self, device, dtype):
        """Verify smooth continuity as nu approaches integer values."""
        self._skip_if_no_scipy()
        x = torch.tensor([2.0], device=device, dtype=dtype)

        for n in [0, 1, 5]:
            # Get exact integer result
            nu_exact = torch.tensor([float(n)], device=device, dtype=dtype)
            I_exact = torch.special.modified_bessel_i(x, nu_exact)
            K_exact = torch.special.modified_bessel_k(x, nu_exact)

            # Check that approaching from above/below gives similar results
            for eps in [1e-6, 1e-4]:
                for sign in [1, -1]:
                    nu_near = torch.tensor([n + sign * eps], device=device, dtype=dtype)
                    I_near = torch.special.modified_bessel_i(x, nu_near)
                    K_near = torch.special.modified_bessel_k(x, nu_near)

                    # Error should be roughly proportional to eps
                    I_rel_err = torch.abs(I_near - I_exact) / torch.abs(I_exact)
                    K_rel_err = torch.abs(K_near - K_exact) / torch.abs(K_exact)

                    # Allow 10x margin for numerical precision
                    self.assertTrue(
                        I_rel_err.item() < 10 * eps,
                        f"I discontinuity at nu={n}: err={I_rel_err.item()}, eps={eps}"
                    )
                    self.assertTrue(
                        K_rel_err.item() < 10 * eps,
                        f"K discontinuity at nu={n}: err={K_rel_err.item()}, eps={eps}"
                    )

    # =========================================================================
    # Gradient tests
    # =========================================================================

    @dtypes(torch.float64)
    def test_modified_bessel_k_gradient(self, device, dtype):
        """Verify gradient formula: dK_nu/dx = -0.5*(K_{nu-1} + K_{nu+1})."""
        x = torch.tensor([1.0, 2.0, 5.0], device=device, dtype=dtype, requires_grad=True)
        nu = torch.tensor([2.5, 2.5, 2.5], device=device, dtype=dtype)

        result = torch.special.modified_bessel_k(x, nu)
        result.sum().backward()

        # Analytical gradient
        with torch.no_grad():
            K_m1 = torch.special.modified_bessel_k(x, nu - 1)
            K_p1 = torch.special.modified_bessel_k(x, nu + 1)
            expected_grad = -0.5 * (K_m1 + K_p1)

        self.assertEqual(x.grad, expected_grad, rtol=1e-4, atol=1e-6)

    @dtypes(torch.float64)
    def test_modified_bessel_i_gradient(self, device, dtype):
        """Verify gradient formula: dI_nu/dx = 0.5*(I_{nu-1} + I_{nu+1})."""
        x = torch.tensor([1.0, 2.0, 5.0], device=device, dtype=dtype, requires_grad=True)
        nu = torch.tensor([2.5, 2.5, 2.5], device=device, dtype=dtype)

        result = torch.special.modified_bessel_i(x, nu)
        result.sum().backward()

        # Analytical gradient
        with torch.no_grad():
            I_m1 = torch.special.modified_bessel_i(x, nu - 1)
            I_p1 = torch.special.modified_bessel_i(x, nu + 1)
            expected_grad = 0.5 * (I_m1 + I_p1)

        self.assertEqual(x.grad, expected_grad, rtol=1e-4, atol=1e-6)

    @dtypes(torch.float64)
    def test_modified_bessel_k_gradcheck(self, device, dtype):
        """Run gradcheck for K_nu."""
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype, requires_grad=True)
        nu = torch.tensor([2.5, 2.5, 2.5], device=device, dtype=dtype)

        from torch.autograd import gradcheck
        def func(x):
            return torch.special.modified_bessel_k(x, nu)

        self.assertTrue(gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3))

    @dtypes(torch.float64)
    def test_modified_bessel_i_gradcheck(self, device, dtype):
        """Run gradcheck for I_nu."""
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype, requires_grad=True)
        nu = torch.tensor([2.5, 2.5, 2.5], device=device, dtype=dtype)

        from torch.autograd import gradcheck
        def func(x):
            return torch.special.modified_bessel_i(x, nu)

        self.assertTrue(gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3))

    # =========================================================================
    # Matern kernel use case (the primary motivation for this implementation)
    # =========================================================================

    @dtypes(torch.float32, torch.float64)
    def test_matern_kernel_use_case(self, device, dtype):
        """Test the Matern kernel use case with nu=12.73."""
        self._skip_if_no_scipy()
        # Matern kernel: K(r) = sigma^2 * (2^{1-nu}/Gamma(nu)) * (sqrt(2*nu)*r/l)^nu * K_nu(sqrt(2*nu)*r/l)
        nu_val = 12.73
        x = torch.linspace(0.01, 20, 200, device=device, dtype=dtype)
        nu = torch.full_like(x, nu_val)

        result = torch.special.modified_bessel_k(x, nu)
        expected = torch.tensor(
            [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
            device=device,
            dtype=dtype
        )

        # Filter out very small values that might have numerical issues
        mask = expected.abs() > 1e-300
        self.assertEqual(result[mask], expected[mask], rtol=1e-5, atol=1e-8)


instantiate_device_type_tests(TestModifiedBesselFunctions, globals())


if __name__ == "__main__":
    run_tests()
