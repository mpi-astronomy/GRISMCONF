"""Test cases for JAX-specific functionality in grismconf.poly module."""

try:
    import jax.numpy as jnp
    import jax.random as jr

    from grismconf.jax import INVPOLY, POLY, npol

    def example(n, m):
        rngkey = jr.PRNGKey(42)  # For reproducibility
        e = jr.uniform(rngkey, (n + 1, npol(m)))
        x = 100.0
        y = 100.0

        d = POLY[(n, m)](e, x, y, 0.0)
        t = INVPOLY[(n, m)](e, x, y, d)
        return d, t

    def test_results():
        test_cases = {
            (2, 1): (0.48870957, 0.0),
            (2, 3): (130.09558, 0.0),
            (2, 6): (16105.361, 0.0),
            (2, 10): (16105.361, 0.0),
        }

        for (n, m), (d_expected, t_expected) in test_cases.items():
            d, t = example(n, m)
            assert jnp.isclose(d, d_expected, atol=1e-7), (
                f"d should be close to {d_expected}"
            )
            assert jnp.isclose(t, t_expected, atol=1e-7), (
                f"t should be close to {t_expected}"
            )

except ImportError:
    print("JAX is not installed. Skipping JAX-specific tests.")

    def test_results():
        print("No tests run due to missing JAX.")
