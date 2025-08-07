"""Jaxified version of grismconf2.py"""

import re
from typing import Mapping, Tuple, override
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array as JDArray

from .grismconf2 import GrismConf
from . import poly
from .poly import npol

POLY = {k: jax.jit(v) for k, v in poly.POLY.items()}
DPOLY = {k: jax.jit(v) for k, v in poly.DPOLY.items()}
INVPOLY = {k: jax.jit(v) for k, v in poly.INVPOLY.items()}


def interp1d(x, y, kind="linear", axis=-1, copy=True, bounds_error=None, fill_value=float("nan"), assume_sorted=False):
    """JAX-compatible 1D interpolation function."""
    if kind != "linear":
        raise NotImplementedError("Only linear interpolation is currently supported in JAX.")

    if not assume_sorted:
        idx = jnp.argsort(x, axis=axis)
        xp = jnp.take_along_axis(x, idx, axis=axis)
        fp = jnp.take_along_axis(y, idx, axis=axis)
    
    @jax.jit
    def interpolator(x_new):
        """Linear interpolation function."""
        return jnp.interp(x_new, xp, fp, left=fill_value, right=fill_value)
    return interpolator


class JaxGrismConf(GrismConf):
    """Jaxified version of GrismConf for JAX compatibility.

    The main difference is to provide the DISP and INVDISP functions as Jax-compatible functions.
    Internal setup remains the same.
    """

    @override
    def _get_disp_data(self) -> Mapping[str, Mapping[str, JDArray]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Extracts the DISP coefficients from the configuration."""
        data = {}
        for key in ("DISPX", "DISPY", "DISPL", "INVDISPX", "INVDISPY", "INVDISPL"):
            data[key] = {}
            for order in self.orders:
                what = re.escape(f"{key}_{order}")
                try:
                    data[key][order] = jnp.vstack(
                        jnp.array(list(self.config[f"{what}.*"].values()))
                    )
                except KeyError:
                    # If no data found for this order, initialize with empty array
                    data[key][order] = jnp.empty((0, 0), dtype=jnp.float32)
        return data

    @override
    def _get_sens_data(self) -> Mapping[str, Tuple[JDArray, JDArray]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Extracts the sensitivity data from the configuration."""
        # get sensitivity data
        data = {}
        for order in self.orders:
            wavelengths, sensitivity = self._get_sensitivity(order)
            data[order] = (jnp.array(wavelengths), jnp.array(sensitivity))
        return data

    @property
    @override
    def POM_POLYGON(self) -> JDArray:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Returns the POM polygon as a JAX-compatible array."""
        return jnp.array(
            [self.config.get("POMX", jnp.nan), self.config.get("POMY", jnp.nan)]
        ).transpose()

    @override
    def rotate_trace(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dx: JDArray,
        dy: JDArray,
        theta: float | None = None,
        origin: Tuple[float, float] = (0.0, 0.0),
    ) -> Tuple[JDArray, JDArray]:
        """Rotate cartesian coordinates CW about an origin

        Parameters
        ----------
        dx, dy : float or `~numpy.ndarray`
            x and y coordinates

        theta : float
            CW rotation angle, in radians

        origin : [float,float]
            Origin about which to rotate

        Returns
        -------
        dxr, dyr : float or `~numpy.ndarray`
            Rotated versions of `dx` and `dy`

        """

        if theta is None:
            theta = self.rotation_theta

        _mat = jnp.array(
            [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
        )

        rot = jnp.dot(jnp.array([dx - origin[0], dy - origin[1]]).T, _mat)
        dxr = rot[:, 0] + origin[0]
        dyr = rot[:, 1] + origin[1]
        return dxr, dyr

    @override
    def DISPL(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        t: JDArray,
    ) -> JDArray:
        """Returns the wavelength corresponding to a value t for an object at position x0,y0

        Parameters
        ----------
        order : str
            Order index

        x0, y0 : float or `~numpy.ndarray`
            x and y coordinates in the direct image

        t : float or `~numpy.ndarray`
            Value of the t variable (usually 0<t<1)

        Returns
        -------
        wav : float or `~numpy.ndarray`
            wavelength value

        """
        return POLY[self._polyname["DISPL"][order]](
            self._disp_data["DISPL"][order], x0, y0, t
        )

    def DDISPL(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        t: JDArray,
    ) -> JDArray:
        """Returns the derivative of the wavelength with respect to t for an object at position x0, y0

        Parameters
        ----------
        order : str
            Order index

        x0, y0 : float or `~numpy.ndarray`
            x and y coordinates in the direct image

        t : float or `~numpy.ndarray`
            Value of the t variable (usually 0<t<1)

        Returns
        -------
        wav : float or `~numpy.ndarray`
            wavelength value

        """
        return DPOLY[self._polyname["DISPL"][order]](
            self._disp_data["DISPL"][order], x0, y0, t
        )

    def DISPX(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        t: JDArray,
    ) -> JDArray:
        """Returns the x offset x'-x = DISPL(x0,y0,t) where x0,y0 is the position on the detector, x'-x is the difference between direct and grism image x-coordinates and 0<t<1

        Parameters
        ----------
        order : str
            Order index

        x0, y0 : float or `~numpy.ndarray`
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        Returns
        -------
        dx : float or `~np.ndarray`
            Trace x-coordinates as a function of `t`

        """
        return -self.wx + POLY[self._polyname["DISPX"][order]](
            self._disp_data["DISPX"][order], x0, y0, t
        )

    def DDISPX(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        t: JDArray,
    ) -> JDArray:
        """Returns the first derivative of the x offset (x'-x) wrt to t, where x0,y0 is the
        position on the detector, x'-x is the difference between direct and grism image x-coordinates and 0<t<1

        Parameters
        ----------
        order : str
            Order index

        x0, y0 : float or `~numpy.ndarray`
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        Returns
        -------
        dxdt : float or `~np.ndarray`
            Trace x-coordinates as a function of `t`

        """
        return DPOLY[self._polyname["DISPX"][order]](
            self._disp_data["DISPX"][order], x0, y0, t
        )

    def DISPY(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        t: JDArray,
    ) -> JDArray:
        """Returns the y offset (y'-y) wrt to t, where x0,y0 is the
        position on the detector, y'-y is the difference between direct and grism image y-coordinates and 0<t<1

        Parameters
        ----------
        order : str
            Order index

        x0, y0 : float or `~numpy.ndarray`
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        Returns
        -------
        dydt : float or `~np.ndarray`
            Trace y-coordinates as a function of `t`

        """
        return -self.wy + POLY[self._polyname["DISPY"][order]](
            self._disp_data["DISPY"][order], x0, y0, t
        )

    def DDISPY(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        t: JDArray,
    ) -> JDArray:
        """Returns the first derivative of the y offset (y'-y) wrt to t, where x0,y0 is the
        position on the detector, y'-y is the difference between direct and grism image y-coordinates and 0<t<1

        Parameters
        ----------
        order : str
            Order index

        x0, y0 : float or `~numpy.ndarray`
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        Returns
        -------
        dydt : float or `~np.ndarray`
            Trace y-coordinates as a function of `t`

        """
        return DPOLY[self._polyname["DISPY"][order]](
            self._disp_data["DISPY"][order], x0, y0, t
        )

    def DISPXY(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        t: JDArray,
        theta: float = 0.0,
    ) -> Tuple[JDArray, JDArray]:
        """Return both `x` and `y` coordinates of a rotated trace

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        theta : float
            CW rotation angle, in radians

        Returns
        -------
        dxr, dyr : float or `~np.ndarray`
            Rotated trace coordinates as a function of `t`

        """
        return (self.DISPX(order, x0, y0, t), self.DISPY(order, x0, y0, t))

        # Rotate coordinates if theta is not zero
        # BUG: ROTATE_COORDS does not exist
        # if np.isclose(theta, 0.):
        #     return dx, dy
        #
        # return self._rotate_coords(dx, dy, theta=theta, origin=[0, 0])

    def INVDISPL(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        wavelength: JDArray,
        t0: JDArray = jnp.linspace(-1, 2, 40),
    ) -> JDArray:
        """Returns the value of 't' that corresponds to a given wavelength for a source at position x0,y0

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        l : float or `~numpy.ndarray`
            Wavelength
        t0 : `~numpy.ndarray`
            Independent variable location where to evaluate the trace.
            For low-order trace shapes, this can be coarsely sampled as
            in the default.

        Returns
        -------
        t : `~numpy.ndarray`
            `t` value

        """
        polyname = self._polyname["DISPL"][order]
        try:
            polydata = self._disp_data["DISPL"][order]
            return INVPOLY[polyname](polydata, x0, y0, wavelength)
        except KeyError:
            invpolydata = self._disp_data["INVDISPL"][order]
            if invpolydata.size > 0:
                return POLY[polyname](invpolydata, x0, y0, wavelength)
            else:
                xr = self.DISPL(order, x0, y0, t0)
                so = jnp.argsort(xr)
                interpolator = interp1d(
                    xr[so],
                    t0[so],
                    bounds_error=False,
                    fill_value="extrapolate",  # type: ignore
                )
                return interpolator(wavelength)

    def INVDISPX(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        dx: JDArray,
        t0: JDArray = jnp.linspace(-1, 2, 40),
    ) -> JDArray:
        """Returns the value of 't' that corresponds to a given x-offset for a source at position x0,y0

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        dx: float or `~numpy.ndarray`
            x-offset between source and a given pixel
        t0 : `~numpy.ndarray`
            Independent variable location where to evaluate the trace.
            For low-order trace shapes, this can be coarsely sampled as
            in the default.

        Returns
        -------
        t : `~numpy.ndarray`
            `t` value

        """
        polyname = self._polyname["DISPX"][order]
        try:
            polydata = self._disp_data["DISPX"][order]
            return INVPOLY[polyname](polydata, x0, y0, dx + self.wx)
        except KeyError:
            invpolydata = self._disp_data["INVDISPX"][order]
            if invpolydata.size > 0:
                return POLY[polyname](invpolydata, x0, y0, dx)
            else:
                xr = self.DISPX(order, x0, y0, t0)
                so = jnp.argsort(xr)
                interpolator = interp1d(
                    xr[so],
                    t0[so],
                )
                return interpolator(dx)

    def INVDISPY(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        order: str,
        x0: JDArray,
        y0: JDArray,
        dy: JDArray,
        t0: JDArray = jnp.linspace(-1, 2, 40),
    ) -> JDArray:
        """Returns the value of 't' that corresponds to a given y-offset for a source at position x0,y0

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        dy: float or `~numpy.ndarray`
            y-offset between source and a given pixel
        t0 : `~numpy.ndarray`
            Independent variable location where to evaluate the trace.
            For low-order trace shapes, this can be coarsely sampled as
            in the default.

        Returns
        -------
        t : `~numpy.ndarray`
            `t` value

        """
        polyname = self._polyname["DISPY"][order]
        try:
            polydata = self._disp_data["DISPY"][order]
            return INVPOLY[polyname](polydata, x0, y0, dy + self.wy)
        except KeyError:
            invpolydata = self._disp_data["INVDISPY"][order]
            if invpolydata.size > 0:
                return POLY[polyname](invpolydata, x0, y0, dy)
            else:
                yr = self.DISPY(order, x0, y0, t0)
                so = jnp.argsort(yr)
                interpolator = interp1d(
                    yr[so],
                    t0[so],
                )
                return interpolator(dy)
