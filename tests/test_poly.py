import numpy as np
from grismconf.poly import npol, POLY, INVPOLY

def example(n, m):
    np.random.seed(42)  # For reproducibility
    e = np.random.rand(n + 1, npol(m))
    x = 100.0
    y = 100.0

    d = POLY[(n, m)](e, x, y, 0.0)
    t = INVPOLY[(n, m)](e, x, y, d)
    return d, t

def test_results():
    test_cases = {
        (2, 1): (0.3745401188473625, 0.0),
        (2, 3): (168.6453649409795, 0.0),
        (2, 6): (9275.361814697739, 0.0),
        (2, 10): (9275.361814697739, 0.0)
    }

    for (n, m), (d_expected, t_expected) in test_cases.items():
        d, t = example(n, m)
        assert np.isclose(d, d_expected, atol=1e-15), f"d should be close to {d_expected}"
        assert np.isclose(t, t_expected, atol=1e-15), f"t should be close to {t_expected}"
