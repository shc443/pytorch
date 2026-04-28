#!/usr/bin/env python3

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import run_tests, TestCase


try:
    from scipy import special as scipy_special

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class TestModifiedBesselFunctions(TestCase):
    def _skip_if_no_scipy(self):
        if not HAS_SCIPY:
            self.skipTest("scipy not available")

    def _tol(self, dtype):
        if dtype == torch.float32:
            return dict(rtol=1e-3, atol=1e-5)
        return dict(rtol=1e-5, atol=1e-8)

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_integer_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        for nu_val in range(11):
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_i(x, nu)
            expected = torch.tensor(
                [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_half_integer_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        for nu_val in [0.5, 1.5, 2.5, 3.5]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_i(x, nu)
            expected = torch.tensor(
                [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_arbitrary_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 15, 50, device=device, dtype=dtype)
        for nu_val in [2.73, 5.17, 12.73]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_i(x, nu)
            expected = torch.tensor(
                [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_edge_cases(self, device, dtype):
        # I_0(0) = 1
        x = torch.tensor([0.0], device=device, dtype=dtype)
        nu = torch.tensor([0.0], device=device, dtype=dtype)
        result = torch.special.modified_bessel_i(x, nu)
        self.assertEqual(result, torch.tensor([1.0], device=device, dtype=dtype))

        # I_nu(0) = 0 for nu > 0
        for nu_val in [1.0, 2.5, 10.0]:
            nu = torch.tensor([nu_val], device=device, dtype=dtype)
            result = torch.special.modified_bessel_i(x, nu)
            self.assertEqual(result, torch.tensor([0.0], device=device, dtype=dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_nan_inf(self, device, dtype):
        nu = torch.tensor([2.5], device=device, dtype=dtype)

        # NaN input propagates
        x_nan = torch.tensor([float("nan")], device=device, dtype=dtype)
        self.assertTrue(torch.isnan(torch.special.modified_bessel_i(x_nan, nu)).all())

        nu_nan = torch.tensor([float("nan")], device=device, dtype=dtype)
        x_ok = torch.tensor([1.0], device=device, dtype=dtype)
        self.assertTrue(
            torch.isnan(torch.special.modified_bessel_i(x_ok, nu_nan)).all()
        )

        # Negative x with non-integer nu returns NaN
        x_neg = torch.tensor([-1.0], device=device, dtype=dtype)
        self.assertTrue(torch.isnan(torch.special.modified_bessel_i(x_neg, nu)).all())

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_negative_x_integer_order(self, device, dtype):
        self._skip_if_no_scipy()
        # I_n(-x) = (-1)^n * I_n(x) for integer n (DLMF 10.27.1)
        x = torch.tensor([-0.5, -1.0, -2.0, -5.0], device=device, dtype=dtype)
        for nu_val in [0, 1, 2, 3, 5, 10]:
            nu = torch.full_like(x, float(nu_val))
            result = torch.special.modified_bessel_i(x, nu)
            expected = torch.tensor(
                [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_integer_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        for nu_val in range(11):
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_k(x, nu)
            expected = torch.tensor(
                [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_half_integer_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        for nu_val in [0.5, 1.5, 2.5, 3.5]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_k(x, nu)
            expected = torch.tensor(
                [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_arbitrary_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 20, 100, device=device, dtype=dtype)
        for nu_val in [2.73, 5.17, 12.73]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_k(x, nu)
            expected = torch.tensor(
                [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_large_nu(self, device, dtype):
        self._skip_if_no_scipy()
        # Regression: nu must not be clamped to 0/inf. The nu > 2000 branch
        # uses the uniform asymptotic expansion (DLMF 10.41) and must match
        # scipy near the x ~ nu crossover where the value is moderate.
        for nu_val, xs in [
            (201.0, [123.7, 250.0, 500.0]),
            (300.0, [123.7, 250.0, 500.0]),
            (2001.0, [1320.66, 1500.0, 1800.0]),
            (2500.0, [1650.0, 2000.0]),
            (5000.0, [3300.0, 4000.0]),
        ]:
            x = torch.tensor(xs, device=device, dtype=dtype)
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_k(x, nu)
            expected = torch.tensor(
                [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            mask = torch.isfinite(expected)
            if mask.any():
                self.assertEqual(result[mask], expected[mask], **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_edge_cases(self, device, dtype):
        # K_nu(0) = inf for any nu
        for nu_val in [0.0, 1.0, 2.5]:
            x = torch.tensor([0.0], device=device, dtype=dtype)
            nu = torch.tensor([nu_val], device=device, dtype=dtype)
            result = torch.special.modified_bessel_k(x, nu)
            self.assertTrue(torch.isinf(result).all())

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_nan_inf(self, device, dtype):
        nu = torch.tensor([2.5], device=device, dtype=dtype)

        # NaN input propagates
        x_nan = torch.tensor([float("nan")], device=device, dtype=dtype)
        self.assertTrue(torch.isnan(torch.special.modified_bessel_k(x_nan, nu)).all())

        nu_nan = torch.tensor([float("nan")], device=device, dtype=dtype)
        x_ok = torch.tensor([1.0], device=device, dtype=dtype)
        self.assertTrue(
            torch.isnan(torch.special.modified_bessel_k(x_ok, nu_nan)).all()
        )

        # Negative x returns NaN
        x_neg = torch.tensor([-1.0], device=device, dtype=dtype)
        self.assertTrue(torch.isnan(torch.special.modified_bessel_k(x_neg, nu)).all())

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_symmetry(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.linspace(0.1, 10, 50, device=device, dtype=dtype)
        nu_pos = torch.full_like(x, 2.73)
        nu_neg = torch.full_like(x, -2.73)

        result_pos = torch.special.modified_bessel_k(x, nu_pos)
        result_neg = torch.special.modified_bessel_k(x, nu_neg)

        # K uses nu = abs(nu) internally, so results should be bitwise equal
        # on float64. Use relaxed tolerance only for float32 rounding.
        if dtype == torch.float64:
            self.assertEqual(result_pos, result_neg, rtol=1e-10, atol=1e-10)
        else:
            self.assertEqual(result_pos, result_neg, rtol=1e-5, atol=1e-5)

    @dtypes(torch.float64)
    def test_modified_bessel_i_near_integer_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([1.0, 2.0, 5.0], device=device, dtype=dtype)

        near_integer_cases = [
            (0.0, 1e-8),
            (0.0, 1e-5),
            (0.0, 0.0001),
            (1.0, 1e-8),
            (1.0, 1e-5),
            (1.0, 0.0001),
            (5.0, 1e-8),
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
                    dtype=dtype,
                )
                self.assertEqual(
                    result,
                    expected,
                    rtol=1e-5,
                    atol=1e-8,
                    msg=f"Failed for nu={nu_val}",
                )

    @dtypes(torch.float64)
    def test_modified_bessel_k_near_integer_orders(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([1.0, 2.0, 5.0], device=device, dtype=dtype)

        near_integer_cases = [
            (0.0, 1e-8),
            (0.0, 1e-5),
            (0.0, 0.0001),
            (1.0, 1e-8),
            (1.0, 1e-5),
            (1.0, 0.0001),
            (5.0, 1e-8),
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
                    dtype=dtype,
                )
                self.assertEqual(
                    result,
                    expected,
                    rtol=1e-5,
                    atol=1e-8,
                    msg=f"Failed for nu={nu_val}",
                )

    @dtypes(torch.float64)
    def test_modified_bessel_continuity_at_integers(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([2.0], device=device, dtype=dtype)

        for n in [0, 1, 5]:
            nu_exact = torch.tensor([float(n)], device=device, dtype=dtype)
            I_exact = torch.special.modified_bessel_i(x, nu_exact)
            K_exact = torch.special.modified_bessel_k(x, nu_exact)

            for eps in [1e-6, 1e-4]:
                for sign in [1, -1]:
                    nu_near = torch.tensor([n + sign * eps], device=device, dtype=dtype)
                    I_near = torch.special.modified_bessel_i(x, nu_near)
                    K_near = torch.special.modified_bessel_k(x, nu_near)

                    I_rel_err = torch.abs(I_near - I_exact) / torch.abs(I_exact)
                    K_rel_err = torch.abs(K_near - K_exact) / torch.abs(K_exact)

                    self.assertTrue(
                        I_rel_err.item() < 10 * eps,
                        f"I discontinuity at nu={n}: err={I_rel_err.item()}, eps={eps}",
                    )
                    self.assertTrue(
                        K_rel_err.item() < 10 * eps,
                        f"K discontinuity at nu={n}: err={K_rel_err.item()}, eps={eps}",
                    )

    @dtypes(torch.float64)
    def test_modified_bessel_k_gradient(self, device, dtype):
        x = torch.tensor(
            [1.0, 2.0, 5.0], device=device, dtype=dtype, requires_grad=True
        )
        nu = torch.tensor([2.5, 2.5, 2.5], device=device, dtype=dtype)

        result = torch.special.modified_bessel_k(x, nu)
        result.sum().backward()

        with torch.no_grad():
            K_m1 = torch.special.modified_bessel_k(x, nu - 1)
            K_p1 = torch.special.modified_bessel_k(x, nu + 1)
            expected_grad = -0.5 * (K_m1 + K_p1)

        self.assertEqual(x.grad, expected_grad, rtol=1e-4, atol=1e-6)

    @dtypes(torch.float64)
    def test_modified_bessel_i_gradient(self, device, dtype):
        x = torch.tensor(
            [1.0, 2.0, 5.0], device=device, dtype=dtype, requires_grad=True
        )
        nu = torch.tensor([2.5, 2.5, 2.5], device=device, dtype=dtype)

        result = torch.special.modified_bessel_i(x, nu)
        result.sum().backward()

        with torch.no_grad():
            I_m1 = torch.special.modified_bessel_i(x, nu - 1)
            I_p1 = torch.special.modified_bessel_i(x, nu + 1)
            expected_grad = 0.5 * (I_m1 + I_p1)

        self.assertEqual(x.grad, expected_grad, rtol=1e-4, atol=1e-6)

    @dtypes(torch.float64)
    def test_modified_bessel_i_gradcheck(self, device, dtype):
        from torch.autograd import gradcheck

        # Wide range: small x, medium x, large x; varied nu
        test_cases = [
            ([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0], 2.5),
            ([1.0, 3.0, 8.0], 0.5),
            ([0.5, 2.0, 7.0], 5.17),
            ([1.0, 5.0, 15.0], 0.01),
        ]
        for x_vals, nu_val in test_cases:
            x = torch.tensor(x_vals, device=device, dtype=dtype, requires_grad=True)
            nu = torch.full_like(x, nu_val).detach()

            def func(x, _nu=nu):
                return torch.special.modified_bessel_i(x, _nu)

            self.assertTrue(
                gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3),
                msg=f"gradcheck failed for nu={nu_val}",
            )

    @dtypes(torch.float64)
    def test_modified_bessel_k_gradcheck(self, device, dtype):
        from torch.autograd import gradcheck

        # Wide range: small x, medium x, large x; varied nu
        test_cases = [
            ([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0], 2.5),
            ([1.0, 3.0, 8.0], 0.5),
            ([0.5, 2.0, 7.0], 5.17),
            ([1.0, 5.0, 15.0], 0.01),
        ]
        for x_vals, nu_val in test_cases:
            x = torch.tensor(x_vals, device=device, dtype=dtype, requires_grad=True)
            nu = torch.full_like(x, nu_val).detach()

            def func(x, _nu=nu):
                return torch.special.modified_bessel_k(x, _nu)

            self.assertTrue(
                gradcheck(func, (x,), eps=1e-6, atol=1e-4, rtol=1e-3),
                msg=f"gradcheck failed for nu={nu_val}",
            )

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_large_x(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([50.0, 100.0, 200.0], device=device, dtype=dtype)
        for nu_val in [0.5, 2.5, 5.0]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_i(x, nu)
            expected = torch.tensor(
                [scipy_special.iv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            # Large x values may have large absolute values; use relative tolerance
            mask = expected.abs() > 0
            if mask.any():
                self.assertEqual(result[mask], expected[mask], **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_large_x(self, device, dtype):
        self._skip_if_no_scipy()
        x = torch.tensor([50.0, 100.0, 200.0], device=device, dtype=dtype)
        for nu_val in [0.5, 2.5, 5.0]:
            nu = torch.full_like(x, nu_val)
            result = torch.special.modified_bessel_k(x, nu)
            expected = torch.tensor(
                [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
                device=device,
                dtype=dtype,
            )
            mask = expected.abs() > 0
            if mask.any():
                self.assertEqual(result[mask], expected[mask], **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_matern_kernel_use_case(self, device, dtype):
        self._skip_if_no_scipy()
        nu_val = 12.73
        x = torch.linspace(0.01, 20, 200, device=device, dtype=dtype)
        nu = torch.full_like(x, nu_val)

        result = torch.special.modified_bessel_k(x, nu)
        expected = torch.tensor(
            [scipy_special.kv(nu_val, xi.item()) for xi in x.cpu()],
            device=device,
            dtype=dtype,
        )

        mask = expected.abs() > 1e-300
        self.assertEqual(result[mask], expected[mask], **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_i_broadcasting(self, device, dtype):
        self._skip_if_no_scipy()
        # x shape (3, 1), nu shape (1, 4) -> (3, 4)
        x = torch.tensor([[1.0], [2.0], [5.0]], device=device, dtype=dtype)
        nu = torch.tensor([[0.5, 1.5, 2.5, 3.5]], device=device, dtype=dtype)
        result = torch.special.modified_bessel_i(x, nu)
        self.assertEqual(result.shape, (3, 4))

        xn, nn = x.cpu().numpy(), nu.cpu().numpy()
        expected = torch.as_tensor(scipy_special.iv(nn, xn), device=device, dtype=dtype)
        self.assertEqual(result, expected, **self._tol(dtype))

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_k_out_parameter(self, device, dtype):
        x = torch.tensor([1.0, 2.0, 3.0], device=device, dtype=dtype)
        nu = torch.full_like(x, 2.5)
        out = torch.empty_like(x)
        ret = torch.special.modified_bessel_k(x, nu, out=out)
        self.assertTrue(ret.data_ptr() == out.data_ptr())
        direct = torch.special.modified_bessel_k(x, nu)
        self.assertEqual(out, direct, **self._tol(dtype))

    def test_modified_bessel_int_to_float_promotion(self, device):
        # int inputs should promote to float (promotes_int_to_float=True in OpInfo)
        x = torch.tensor([1, 2, 3], device=device, dtype=torch.int64)
        nu = torch.tensor([1, 2, 3], device=device, dtype=torch.int64)
        result = torch.special.modified_bessel_i(x, nu)
        self.assertTrue(result.is_floating_point())

    @dtypes(torch.float32, torch.float64)
    def test_modified_bessel_cpu_cuda_parity(self, device, dtype):
        # Verify CPU and CUDA produce equivalent results at boundary inputs
        # where the type-aware constants in Math.h / Math.cuh matter.
        if device == "cpu" or not torch.cuda.is_available():
            self.skipTest("requires CUDA for cross-device comparison")

        # Boundary inputs covering: small x (series), medium x (Temme/CF2),
        # large x (asymptotic), and large nu (UAE branch).
        x_cpu = torch.tensor(
            [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0, 1000.0],
            dtype=dtype
        )
        x_cuda = x_cpu.to(device)

        for nu_val in [0.5, 2.5, 12.73, 50.0, 200.0, 2001.0, 5000.0]:
            nu_cpu = torch.full_like(x_cpu, nu_val)
            nu_cuda = nu_cpu.to(device)

            for fn_name in ("modified_bessel_i", "modified_bessel_k"):
                fn = getattr(torch.special, fn_name)
                out_cpu = fn(x_cpu, nu_cpu)
                out_cuda = fn(x_cuda, nu_cuda).cpu()

                # Compare on inputs where both are finite and non-zero
                mask = (
                    torch.isfinite(out_cpu)
                    & torch.isfinite(out_cuda)
                    & (out_cpu.abs() > 1e-300)
                )
                if not mask.any():
                    continue

                rel_err = (
                    (out_cpu[mask] - out_cuda[mask]).abs() / out_cpu[mask].abs()
                ).max().item()
                # Tight bound: post-refactor CPU/CUDA agreement should be
                # within a few ULP for both float32 and float64.
                tol = 1e-5 if dtype == torch.float32 else 1e-12
                self.assertLess(
                    rel_err, tol,
                    msg=f"{fn_name} nu={nu_val} dtype={dtype}: "
                        f"CPU/CUDA rel_err={rel_err:.2e} exceeds {tol:.0e}",
                )


instantiate_device_type_tests(TestModifiedBesselFunctions, globals())


if __name__ == "__main__":
    run_tests()
