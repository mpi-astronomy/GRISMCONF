import os
import re
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from scipy.interpolate import interp1d

from . import poly, specwcs


class interp1d_picklable:
    """class wrapper for piecewise linear function

    # FIXME: Do we actually need this?
    """

    def __init__(self, xi, yi, **kwargs):
        self.xi = xi
        self.yi = yi
        self.kwargs = kwargs
        self.f = interp1d(xi, yi, **kwargs)

    def __call__(self, xnew):
        return self.f(xnew)

    def __getstate__(self):
        return self.xi, self.yi, self.kwargs

    def __setstate__(self, state):
        self.f = interp1d(state[0], state[1], **state[2])


class Config:
    """Class to read and hold GRISM configuration info

    # FIXME: Refactor this class for 2 classes
    #        This class changes API significantly between the two modes
    #        1. Using a datamodel (e.g., from a fits file)
    #        2. Using a grismconf file (e.g., from a text file)
    #        This class should be split into two classes, one for each mode fine
    #        to have a parent class for common API.
    #
    #        It could be that not all attributes are necessary in the main API.
    #
    """

    def __init__(self, filename, DIRFILTER=None):
        self._DISPX_data = {}
        self._DISPY_data = {}
        self._DISPL_data = {}

        self._INVDISPX_data = {}
        self._INVDISPY_data = {}
        self._INVDISPL_data = {}

        self._DISPX_polyname = {}
        self._DISPY_polyname = {}
        self._DISPL_polyname = {}

        self._INVDISPX_polyname = {}
        self._INVDISPY_polyname = {}
        self._INVDISPL_polyname = {}

        self.SENS = {}
        self.SENS_data = {}

        # Extent of FOV in detector pixel
        self.XRANGE = {}
        self.YRANGE = {}

        # Wavelength range of the grism
        self.WRANGE = {}

        self.orders = []

        self.wx: float = 0.0
        self.wy: float = 0.0

        try:
            # fits.open(filename)  # file type check... is this necessary?
            # Annoyingly testing for fits with a large library
            # Replacing by suffix testing (which is what jwst datamodel does anyways)
            if Path(filename).suffix.lower() != ".fits":
                raise OSError("File is not a FITS file")
            self.__init_DATAMODEL(filename)
        except OSError:
            self.__init_GRISMCONF(filename, DIRFILTER=None)

    def __init_DATAMODEL(self, filename: str) -> None:
        print(f"Loading from datamodel of {filename}")

        self._DISPX_data, self._DISPY_data, self._DISPL_data, self.SENS_data = (
            specwcs.specwcs_poly(filename)
        )
        self.orders = list(self._DISPX_data.keys())
        self.wx = 0.0
        self.wy = 0.0

        for order in self.orders:
            # self.SENS[order] = self._get_sensitivity(order)

            self._DISPX_polyname[order] = np.shape(self._DISPX_data[order])
            self._DISPY_polyname[order] = np.shape(self._DISPY_data[order])
            self._DISPL_polyname[order] = np.shape(self._DISPL_data[order])

            # self.SENS_data[order] = self._get_sensitivity(order)

            vg = self.SENS_data[order][1] > np.max(self.SENS_data[order][1]) * 1e-3
            wmin = np.min(self.SENS_data[order][0][vg])
            wmax = np.max(self.SENS_data[order][0][vg])
            self.WRANGE[order] = [wmin, wmax]

            self.SENS[order] = interp1d_picklable(
                self.SENS_data[order][0],
                self.SENS_data[order][1],
                bounds_error=False,
                fill_value=0.0,
            )

    def __init_GRISMCONF(self, GRISM_CONF: str, DIRFILTER: str | None = None) -> None:
        """Return a Config object

        Parameters
        ----------
        GRISM_CONF : str
            The full path and name to a grism configuration file

        DIRFILTER : str
            The name of the direct filter so that filter wedge offsets can be included.
            Should match the filter used when a direct image was used in the same visit as
            the grism observations
        """

        self.GRISM_CONF = open(GRISM_CONF).readlines()
        self.GRISM_CONF_PATH = os.path.dirname(GRISM_CONF)
        self.GRISM_CONF_FILE = os.path.basename(GRISM_CONF)

        # Extent of FOV in detector pixel
        self.XRANGE = {}
        self.YRANGE = {}

        self.rotation_theta = 0.0
        self.FWCPOS_REF = None
        self.POM = None
        self.POMX = None
        self.POMY = None
        self.POM_POLYGON = None
        self.BCK = None

        # Get grism orders from the configuration file
        self.orders = [
            line.strip().split()[0].split("_")[-1]
            for line in self.GRISM_CONF
            if line.startswith("BEAM_")
        ]

        if DIRFILTER is not None:
            # We get the wedge offset values for this direct filter
            r = self._get_value(f"WEDGE_{DIRFILTER}", type=float)
            self.wx = r[0]
            self.wy = r[1]
        else:
            self.wx = 0.0
            self.wy = 0.0

        # Get physical size of detector
        self.NAXIS = self._get_value("NAXIS", type=int)

        try:
            self.FWCPOS_REF = float(self._get_value("FWCPOS_REF"))
        except Exception:
            pass

        # Load the name of a POM file
        try:
            self.POM = os.path.join(self.GRISM_CONF_PATH, self._get_value("POM"))
        except Exception:
            pass

        # Load POMX and POMY polynomials if they are specified
        try:
            self.POMX = np.array(self._get_value("POMX")).astype(float)
        except Exception:
            pass

        try:
            self.POMY = np.array(self._get_value("POMY")).astype(float)
        except Exception:
            pass

        if self.POMX is not None and self.POMY is not None:
            if np.isfinite(self.POMX).all() and np.isfinite(self.POMY).all():
                self.POM_POLYGON = np.array([self.POMX, self.POMY]).transpose()

        # Load the name of a dispersed background model file
        try:
            self.BCK = os.path.join(self.GRISM_CONF_PATH, self._get_value("BACKGROUND"))
        except Exception:
            pass

        for order in self.orders:
            self._DISPX_data[order] = self._get_parameters("DISPX", order)
            self._DISPY_data[order] = self._get_parameters("DISPY", order)
            self._DISPL_data[order] = self._get_parameters("DISPL", order)
            self.SENS[order] = self._get_sensitivity(order)

            self._DISPX_polyname[order] = np.shape(self._DISPX_data[order])
            self._DISPY_polyname[order] = np.shape(self._DISPY_data[order])
            self._DISPL_polyname[order] = np.shape(self._DISPL_data[order])

            self.SENS_data[order] = self._get_sensitivity(order)

            self._INVDISPX_data[order] = self._get_parameters("INVDISPX", order)
            self._INVDISPY_data[order] = self._get_parameters("INVDISPY", order)
            self._INVDISPL_data[order] = self._get_parameters("INVDISPL", order)

            self._INVDISPX_polyname[order] = np.shape(self._INVDISPX_data[order])
            self._INVDISPY_polyname[order] = np.shape(self._INVDISPY_data[order])
            self._INVDISPL_polyname[order] = np.shape(self._INVDISPL_data[order])

            vg = self.SENS_data[order][1] > np.max(self.SENS_data[order][1]) * 1e-3
            wmin = np.min(self.SENS_data[order][0][vg])
            wmax = np.max(self.SENS_data[order][0][vg])
            self.WRANGE[order] = [wmin, wmax]

            self.SENS[order] = interp1d_picklable(
                self.SENS_data[order][0],
                self.SENS_data[order][1],
                bounds_error=False,
                fill_value=0.0,
            )

            self.XRANGE[order] = self._get_value("XRANGE_%s" % (order), type=float)
            self.YRANGE[order] = self._get_value("YRANGE_%s" % (order), type=float)

    def repickle_sens(self, order):
        self.SENS[order] = interp1d_picklable(
            self.SENS_data[order][0],
            self.SENS_data[order][1],
            bounds_error=False,
            fill_value=0.0,
        )

    def set_rotation(self, fwcpos=None):
        if fwcpos is not None:
            # print(self.FWCPOS_REF,fwcpos)
            self.rotation_theta = np.radians(fwcpos - self.FWCPOS_REF)

    def rotate_trace(self, dx, dy, theta=None, origin=[0, 0]):
        """Rotate cartesian coordinates CW about an origin

        Parameters
        ----------
        dx, dy : float or `~numpy.ndarray`
            x and y coordinages

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
        _mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        rot = np.dot(np.array([dx - origin[0], dy - origin[1]]).T, _mat)
        dxr = rot[:, 0] + origin[0]
        dyr = rot[:, 1] + origin[1]
        return dxr, dyr

    def is_inside_POM(self, order, xs, ys, XRANGE=False):
        """Check if points xs,ys are within the POM. Uses self.POM_POLYGON is availanle, otherwise XRANGE,YRANGE"""
        # print("is_inside:",self.POM_POLYGON)
        if self.POM_POLYGON is not None and XRANGE is not True:
            # print("use polygon")
            import matplotlib.path as mpltPath

            points = np.array([xs, ys]).transpose()
            path = mpltPath.Path(self.POM_POLYGON)
            ok = path.contains_points(points)
            return ok
        if self.XRANGE[order] is not None and self.YRANGE[order] is not None:
            # print("use xrange")
            xs = np.array(xs)
            ys = np.array(ys)
            xminus = self.XRANGE[order][0]
            xplus = self.XRANGE[order][1]
            yminus = self.YRANGE[order][0]
            yplus = self.YRANGE[order][1]

            ok = xs < self.NAXIS[1] + xplus
            ok = ok & (xs > xminus)
            ok = ok & (ys < self.NAXIS[0] + yplus)
            ok = ok & (ys > yminus)
            return ok

    def DISPL(self, order, x0, y0, t):
        """Returns the wavelength corresponding to a value t for an object at posittion x0,y0

        Parameters
        ----------
        x0, y0 : float or `~numpy.ndarray`
            x and y coordinates in the direct image

        t : float
            Value of the t variable (usually 0<t<1)

        Returns
        -------
        wav : `~numpy.ndarray`
            wavelength value

        """
        return poly.POLY[self._DISPL_polyname[order]](
            self._DISPL_data[order], x0, y0, t
        )

    def DDISPL(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Returns the first derivate of the wavelength (wrt to t) for a value t for an object at posittion x0,y0

        Parameters
        ----------
        x0, y0 : float or `~numpy.ndarray`
            x and y coordinates in the direct image

        t : float
            Value of the t variable (usually 0<t<1)

        Returns
        -------
        dwav : `~numpy.ndarray`
            First derivative of the wavelength with respect to 't', as a function of 't'
        """
        return poly.DPOLY[self._DISPL_polyname[order]](
            self._DISPL_data[order], x0, y0, t
        )

    def DISPXY(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
        theta: float = 0,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        dx = -self.wx + poly.POLY[self._DISPX_polyname[order]](
            self._DISPX_data[order], x0, y0, t
        )
        dy = -self.wy + poly.POLY[self._DISPY_polyname[order]](
            self._DISPY_data[order], x0, y0, t
        )

        if theta != 0:
            # BUG: _rotate_coords does not exist
            dxr, dyr = self._rotate_coords(dx, dy, theta=theta, origin=[0, 0])
            return dxr, dyr
        else:
            return dx, dy

    def INVDISPXY(
        self, order, x0, y0, dx=None, dy=None, theta=0, t0=np.linspace(-1, 2, 40)
    ):
        """Return independent variable `t` along rotated trace

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        dx : float, `~numpy.ndarray` or None
            `x` coordinate in *rotated* trace where to evaluate the trace
            independent variable `t`.

        dy : float, `~numpy.ndarray` or None
            Same as `dx` but evaluate along 'y' axis.

        t0 : `~np.ndarray`
            Independent variable location where to evaluate the rotated trace.
            For low-order trace shapes, this can be coarsely sampled as
            in the default.

        Returns
        -------
        tr : float or `~np.ndarray`
            Independent variable `t` evaluated on the rotated trace at
            `dx` or `dy`.

        .. note::

        Order of execution is first check if `dx` supplied.  If not, then
        check `dy`.  And if both are None, then return None (do nothing).

        """
        if dx is not None:
            xr, yr = self.DISPXY(order, x0, y0, t0, theta=theta)
            so = np.argsort(xr)
            f = interp1d_picklable(xr[so], t0[so])
            tr = f(dx)
            return tr

        if dy is not None:
            xr, yr = self.DISPXY(order, x0, y0, t0, theta=theta)
            so = np.argsort(yr)
            f = interp1d_picklable(yr[so], t0[so])
            tr = f(dy)
            return tr

        return None

    def DISPX(self, order, x0, y0, t):
        """Returns the x offset x'-x = DISPL(x0,y0,t) where x0,y0 is the
        position on the detector, x'-x is the difference between direct and grism image x-coordinates and 0<t<1

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        Returns
        -------
        dx : float or `~np.ndarray`
            Trace x-coordinates as a function of `t`

        """
        dx = -self.wx + poly.POLY[self._DISPX_polyname[order]](
            self._DISPX_data[order], x0, y0, t
        )

        return dx

    def DDISPX(self, order, x0, y0, t):
        """Returns the first derivative of the x offset (x'-x) wrt to t, where x0,y0 is the
        position on the detector, x'-x is the difference between direct and grism image x-coordinates and 0<t<1

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        Returns
        -------
        dxdt : float or `~np.ndarray`
            First derivative of the trace x-coordinates with respect to 't', as a function of `t`

        """
        return poly.DPOLY[self._DISPX_polyname[order]](
            self._DISPX_data[order], x0, y0, t
        )

    def DISPY(self, order, x0, y0, t):
        """Returns the x offset (y'-y) wrt to t, where x0,y0 is the
        position on the detector, y'-y is the difference between direct and grism image y-coordinates and 0<t<1

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        Returns
        -------
        dydt : float or `~np.ndarray`
            First derivative of the trace y-coordinates with respect to 't', as a function of `t`

        """
        return -self.wy + poly.POLY[self._DISPY_polyname[order]](
            self._DISPY_data[order], x0, y0, t
        )

    def DDISPY(self, order, x0, y0, t):
        """Returns the first derivative of the y offset (y'-y) wrt to t, where x0,y0 is the
        position on the detector, y'-y is the difference between direct and grism image x-coordinates and 0<t<1

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        t : float or `~numpy.ndarray`
            Parameter where to evaluate the trace

        Returns
        -------
        dydt : float or `~np.ndarray`
            First derivative of the trace y-coordinates with respect to 't', as a function of `t`

        """
        return poly.DPOLY[self._DISPY_polyname[order]](
            self._DISPY_data[order], x0, y0, t
        )

    def INVDISPL(self, order, x0, y0, l, t0=np.linspace(-1, 2, 40)):
        """Returns the value of 't' that corresponds to a given wavelength for a source at position x0,y0

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        l : float or `~numpy.ndarray`
            Wavelength
        t0 : `~np.ndarray`
            Independent variable location where to evaluate the trace.
            For low-order trace shapes, this can be coarsely sampled as
            in the default.
        Returns
        -------
        t : float or `~np.ndarray`
            `t` value

        """

        if self._DISPL_polyname[order] in poly.INVPOLY.keys():
            return poly.INVPOLY[self._DISPL_polyname[order]](
                self._DISPL_data[order], x0, y0, l
            )
        elif len(self._INVDISPL_data[order]) == 0:
            xr = self.DISPL(order, x0, y0, t0)
            so = np.argsort(xr)
            f = interp1d_picklable(
                xr[so], t0[so], bounds_error=False, fill_value="extrapolate"
            )
            tr = f(l)
            return tr
        else:
            return poly.POLY[self._INVDISPL_polyname[order]](
                self._INVDISPL_data[order], x0, y0, l
            )

    def INVDISPX(self, order, x0, y0, dx, t0=np.linspace(-1, 2, 40)):
        """Returns the value of 't' that corresponds to a given x-offset for a source at position x0,y0

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        dx: float or `~numpy.ndarray`
            x-offset between source and a given pixel
        t0 : `~np.ndarray`
            Independent variable location where to evaluate the trace.
            For low-order trace shapes, this can be coarsely sampled as
            in the default.
        Returns
        -------
        t : float or `~np.ndarray`
            `t` value

        """
        if self._DISPX_polyname[order] in poly.INVPOLY.keys():
            return poly.INVPOLY[self._DISPX_polyname[order]](
                self._DISPX_data[order], x0, y0, dx + self.wx
            )
        elif len(self._INVDISPX_data[order]) == 0:
            xr = self.DISPX(order, x0, y0, t0)
            so = np.argsort(xr)
            f = interp1d_picklable(xr[so], t0[so])
            tr = f(dx)
            return tr
        else:
            return poly.POLY[self._INVDISPX_polyname[order]](
                self._INVDISPX_data[order], x0, y0, dx
            )

    def INVDISPY(self, order, x0, y0, dy, t0=np.linspace(-1, 2, 40)):
        """Returns the value of 't' that corresponds to a given y-offset for a source at position x0,y0

        Parameters
        ----------
        order : str
            Order string

        x0, y0 : float
            Reference position (i.e., in direct image)

        dy: float or `~numpy.ndarray`
            y-offset between source and a given pixel
        t0 : `~np.ndarray`
            Independent variable location where to evaluate the trace.
            For low-order trace shapes, this can be coarsely sampled as
            in the default.
        Returns
        -------
        t : float or `~np.ndarray`
            `t` value

        """
        if self._DISPY_polyname[order] in poly.INVPOLY.keys():
            return poly.INVPOLY[self._DISPY_polyname[order]](
                self._DISPY_data[order], x0, y0, dy + self.wy
            )
        elif len(self._INVDISPY_data[order]) == 0:
            xr, yr = self.DISPXY(order, x0, y0, t0)
            so = np.argsort(yr)
            f = interp1d_picklable(yr[so], t0[so])
            tr = f(dy)
            return tr
        else:
            return poly.POLY[self._INVDISPY_polyname[order]](
                self._INVDISPY_data[order], x0, y0, dy
            )

    @staticmethod
    def _get_grism_orders(GRISM_CONF: list[str]) -> list[str]:
        """Returns all the know orders in Config

        Parameters
        ----------

        Returns
        -------
        orders: `array`
            List of orders

        """

        orders = []
        # Get orders
        for line in GRISM_CONF:
            k = "BEAM_"
            if line[0 : len(k)] == k:
                ws = line.split()
                order = ws[0].split("_")[-1]
                orders.append(order)
        return orders

    def _get_parameters(self, name, order, str_fmt="%s_%s_"):
        """Return the 2D polynomial array stored in the config file"""
        str = str_fmt % (name, order)
        # Find out how many we have to store
        n = 0
        m = 0
        for l in self.GRISM_CONF:
            if l[0] == "#":
                continue
            ws = l.split()
            if len(ws) > 0 and str == ws[0][0 : len(str)]:
                i = ws[0].split(str)[-1]
                n = n + 1
                m = len(ws) - 1

        arr = np.zeros((n, m))

        for l in self.GRISM_CONF:
            ws = l.split()
            if len(ws) > 0 and str == ws[0][0 : len(str)]:
                i = int(ws[0].split(str)[-1])
                if len(ws) - 1 != m:
                    print("Wrong format for ", self.GRISM_CONF, name, order)
                    sys.exit(10)
                vals = [float(ww) for ww in ws[1:]]
                arr[i, 0:m] = vals

        return arr

    def _get_value(self, key: str, type: Callable | None = None):
        """Helper function to simply return the value for a simple keyword parameters
        in the config file."""

        for l in self.GRISM_CONF:
            ws = l.split()
            if len(ws) > 0 and ws[0] == key:
                if len(ws) == 2:
                    if type == None:
                        return ws[1]
                    elif type == float:
                        return float(ws[1])
                    elif type == int:
                        return int(ws[1])
                else:
                    if type == None:
                        return ws[1:]
                    elif type == float:
                        return [float(x) for x in ws[1:]]
                    elif type == int:
                        return [int(x) for x in ws[1:]]
        return None

    def _get_sensitivity(self, order: str):
        """Helper function that looks for the name of the sensitivity file, reads it and
        stores the content in a simple list [WAVELENGTH, SENSITIVITY]."""
        fname = os.path.join(
            self.GRISM_CONF_PATH,  # type: ignore
            self._get_value("SENSITIVITY_%s" % (order)),  # type: ignore[call-arg] # noqa: E501
        )
        with fits.open(fname) as fin:
            wavs = fin[1].data.field("WAVELENGTH")[:] * 1
            sens = fin[1].data.field("SENSITIVITY")[:] * 1

        # Fix for cases where sensitivity is not zero on edges
        sens[0:2] = 0.0
        sens[-2:] = 0.0

        return [wavs, sens]


class ConfigDatamodel(Config):
    def __init__(self, filename: str, **kwargs):
        """Initialize ConfigDatamodel object from a FITS file."""
        self._DISPX_data = {}
        self._DISPY_data = {}
        self._DISPL_data = {}

        self._INVDISPX_data = {}
        self._INVDISPY_data = {}
        self._INVDISPL_data = {}

        self._DISPX_polyname = {}
        self._DISPY_polyname = {}
        self._DISPL_polyname = {}

        self._INVDISPX_polyname = {}
        self._INVDISPY_polyname = {}
        self._INVDISPL_polyname = {}

        self.SENS = {}
        self.SENS_data = {}

        # Extent of FOV in detector pixel
        self.XRANGE = {}
        self.YRANGE = {}

        # Wavelength range of the grism
        self.WRANGE = {}

        self.orders = []

        self.wx: float = 0.0
        self.wy: float = 0.0

        if Path(filename).suffix.lower() != ".fits":
            raise OSError("File is not a FITS file")
        self.__init_DATAMODEL(filename)


class GrismconfParser(dict):
    """ " Parser for grism configuration files into a dictionary like object.

    This dictionary object can be used to access multiple keys at onces using pattern matching.

    Attributes
    ----------
    mapper : dict
        A dictionary mapping configuration keys to their respective data types.
        This is used to convert string values in the configuration file to their
        appropriate types when parsing the file.
        defaults to a set of common keys and types, but can be extended /
        overwritten with `mapper` keyword arguments.

    Methods
    -------
    from_content(content: str | list[str], **kwargs) -> dict
        Parses the content of a grism configuration file from a string or list of strings.
        Returns a dictionary with the parsed configuration values.
    from_file(filename: str, **kwargs) -> dict
        Reads a grism configuration file from disk and parses its content.
        Returns a dictionary with the parsed configuration values.
        (Raises OSError if the file does not exist or is not a file)
    """

    __mapper__: dict[str, Callable | type] = {
        "NAXIS": int,
        "DISPL": float,
        "DISPY": float,
        "DISPX": float,
        "XRANGE": float,
        "YRANGE": float,
        "POMX": float,
        "POMY": float,
        "WEDGE": float,
        "FWCPOS_REF": float,
    }

    def __getitem__(self, key: Any) -> Any:
        """Get the value(s) associated with a key or pattern in the configuration."""
        if value := super().get(key):
            return value
        # not direct match, try pattern search
        matcher = re.compile(rf"{key}")
        matches = self.__class__({k: v for k, v in self.items() if matcher.match(k)})
        if matches:
            return matches
        raise KeyError(f"Key '{key}' not found in configuration.")

    @classmethod
    def from_content(cls, content: str | list[str], **kwargs):
        """Factory method to create a Config object from a string."""
        if isinstance(content, str):
            content = content.strip().split("\n")

        mapper = cls.__mapper__.copy()
        mapper.update(kwargs.get("mapper", {}))

        data = {}

        for line in content:
            if line.startswith("#"):
                continue

            elements = line.strip().split()

            if not elements:
                continue

            # <name>_<+order>_j should still point to <name>
            value_mapper = mapper.get(elements[0].split("_")[0], lambda x: x)
            values = [value_mapper(k) for k in elements[1:]]
            if not values:
                data[elements[0]] = None
            elif len(values) == 1:
                data[elements[0]] = values[0]
            else:
                data[elements[0]] = values

        return cls(data)

    @classmethod
    def from_file(cls, filename: str, **kwargs):
        """Factory method to create a Config object from a file."""

        path = Path(filename)

        if not (path.exists() and path.is_file()):
            raise OSError(f"File {filename} does not exist or is not a file.")

        with path.open("r") as f:
            content = f.readlines()
        if not content:
            raise ValueError(f"File {filename} is empty.")

        return cls.from_content(content, **kwargs)


class GrismConf:
    """Class to read and hold GRISM configuration info from a grismconf file"""

    def __init__(self, filename: str, dirfilter: str | None = None):
        """Initialize GrismConf object from a grismconf file."""

        # Internals
        self.source: Path = Path(filename)
        self.config: GrismconfParser = GrismconfParser.from_file(filename)

        # Get grism orders from the configuration file
        self.orders: list[str] = [
            k.split("_")[-1] for k in self.config["BEAM_*"].keys()
        ]

        # dirfilter --> wx, wy
        self.wx: float = 0.0
        self.wy: float = 0.0
        self.set_dirfilter(dirfilter)

        # Rotation angle
        self.rotation_theta: float = 0.0

        # get pixel (DISP & INVDISP) coefficients
        # note: when missing defined as array([], shape=(0, 0), dtype=float64
        self._disp_data: dict[str, dict[str, np.ndarray]] = self._get_disp_data()

        # shapes of the DISP and INVDISP coefficients per order
        self._polyname = {}
        for key in self._disp_data.keys():
            self._polyname[key] = {}
            for order in self.orders:
                self._polyname[key][order] = np.shape(self._disp_data[key][order])

        # get sensitivity data
        self._sens_data: dict[str, Tuple[np.ndarray, np.ndarray]] = (
            self._get_sens_data()
        )

        # set wavelength range for each order
        self.WRANGE: dict[str, Tuple[float, float]] = self._set_wrange(self._sens_data)

    def get_sensitivity_function(self, order: str) -> interp1d:
        """Get the sensitivity function for a specific order.

        TODO: Provide a jax-compatible version of this interp1d function.
        """
        if order not in self._sens_data:
            raise ValueError(f"Sensitivity data for order '{order}' not found.")
        wavs, sens = self._sens_data[order]
        return interp1d(wavs, sens, bounds_error=False, fill_value=0.0)

    def set_dirfilter(self, dirfilter: str | None = None):
        """Set the direct filter for this grism configuration."""
        if dirfilter is not None:
            # We get the wedge offset values for this direct filter
            wx, wy = self.config[f"WEDGE_{dirfilter}"]
        else:
            wx, wy = 0.0, 0.0

        self.wx = wx
        self.wy = wy

    def set_rotation(self, fwcpos: float | None = None):
        """Set the rotation angle based on the FWC position."""
        if fwcpos is None:
            return
        self.rotation_theta = np.radians(fwcpos - self.FWCPOS_REF)  # type: ignore

    def _get_disp_data(self) -> dict[str, dict[str, np.ndarray]]:
        """Extracts the DISP coefficients from the configuration."""
        data = {}
        for key in ("DISPX", "DISPY", "DISPL", "INVDISPX", "INVDISPY", "INVDISPL"):
            data[key] = {}
            for order in self.orders:
                what = re.escape(f"{key}_{order}")
                try:
                    data[key][order] = np.vstack(
                        list(self.config[f"{what}.*"].values())
                    )
                except KeyError:
                    # If no data found for this order, initialize with empty array
                    data[key][order] = np.empty((0, 0))
        return data

    def _get_sens_data(self) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Extracts the sensitivity data from the configuration."""
        # get sensitivity data
        data = {}
        for order in self.orders:
            data[order] = self._get_sensitivity(order)
        return data

    @staticmethod
    def _set_wrange(
        sens_data: dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, Tuple[float, float]]:
        """Set the wavelength range for each order based on the sensitivity data."""
        wrange = {}
        for order, (wavs, sens) in sens_data.items():
            vg = sens > np.max(sens) * 1e-3
            wmin = np.min(wavs[vg])
            wmax = np.max(wavs[vg])
            wrange[order] = (wmin, wmax)
        return wrange

    def _get_file_path(self, key: str) -> str:
        """Return the full path for a file specified in the configuration."""
        if key in self.config:
            return os.path.join(self.source.parent, self.config[key])
        else:
            raise ValueError(f"{key} not defined in the configuration file.")

    def _get_sensitivity(self, order: str) -> Tuple[np.ndarray, np.ndarray]:
        """Helper function that looks for the name of the sensitivity file,
        reads it and stores the content in a simple list
        [WAVELENGTH, SENSITIVITY].
        """
        fname = self._get_file_path(f"SENSITIVITY_{order}")

        with fits.open(fname) as fin:
            wavs = fin[1].data.field("WAVELENGTH")[:] * 1
            sens = fin[1].data.field("SENSITIVITY")[:] * 1

        # Fix for cases where sensitivity is not zero on edges
        sens[0:2] = 0.0
        sens[-2:] = 0.0

        return wavs, sens

    @property
    def XRANGE(self) -> dict:
        return {
            k.replace("XRANGE_", ""): v for k, v in self.config["XRANGE_.*"].items()
        }

    @property
    def YRANGE(self) -> dict:
        return {
            k.replace("YRANGE_", ""): v for k, v in self.config["YRANGE_.*"].items()
        }

    @property
    def BCK(self) -> str | None:
        """Return the background model file name."""
        try:
            return self._get_file_path("BACKGROUND")
        except ValueError:
            # If no background is specified, return None
            return None

    @property
    def POM(self) -> str | None:
        """Return the POM file name."""
        try:
            return self._get_file_path("POM")
        except ValueError:
            # If no POM is specified, return None
            return None

    @property
    def NAXIS(self) -> Tuple[int, int]:
        """Return the physical size of the detector."""
        if "NAXIS" in self.config:
            return tuple(self.config["NAXIS"])
        else:
            raise ValueError("NAXIS not defined in the configuration file.")

    @property
    def POM_POLYGON(self) -> np.ndarray:
        return np.array(
            [self.config.get("POMX", np.nan), self.config.get("POMY", np.nan)]
        ).transpose()

    @property
    def POMX(self) -> float:
        return self.config.get("POMX", np.nan)

    @property
    def POMY(self) -> float:
        return self.config.get("POMY", np.nan)

    @property
    def FWCPOS_REF(self) -> float | None:
        """Return the reference position for the FWC."""
        return self.config.get("FWCPOS_REF", None)

    @property
    def _DISPL_data(self) -> dict:
        return self._disp_data["DISPL"]

    @property
    def _DISPX_data(self) -> dict:
        return self._disp_data["DISPX"]

    @property
    def _DISPY_data(self) -> dict:
        return self._disp_data["DISPY"]

    def rotate_trace(
        self,
        dx: float | npt.NDArray[np.float64],
        dy: float | npt.NDArray[np.float64],
        theta: float | None = None,
        origin: Tuple[float, float] = (0, 0),
    ) -> Tuple[float | npt.NDArray[np.float64], float | npt.NDArray[np.float64]]:
        """Rotate cartesian coordinates CW about an origin

        Parameters
        ----------
        dx, dy : float or `~numpy.ndarray`
            x and y coordinages

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

        _mat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        rot = np.dot(np.array([dx - origin[0], dy - origin[1]]).T, _mat)
        dxr = rot[:, 0] + origin[0]
        dyr = rot[:, 1] + origin[1]
        return dxr, dyr

    def is_inside_POM(
        self,
        order: int,
        xs: npt.NDArray[np.float64],
        ys: npt.NDArray[np.float64],
        XRANGE: bool = False,
    ) -> npt.NDArray[np.bool_]:
        """Check if points xs,ys are within the POM. Uses self.POM_POLYGON is availaible, otherwise XRANGE,YRANGE"""

        # Use polygon path from matplotlib if available otherwise based on X,Y RANGE
        if np.isfinite(self.POM_POLYGON).all() and not XRANGE:
            import matplotlib.path as mpltPath

            path = mpltPath.Path(self.POM_POLYGON)
            points = np.array([xs, ys]).transpose()
            return path.contains_points(points)

        xrange = self.XRANGE.get(order)
        yrange = self.YRANGE.get(order)
        if xrange is None or yrange is None:
            raise ValueError(f"XRANGE or YRANGE not defined for order {order}.")
        xs = np.array(xs)
        ys = np.array(ys)

        xminus, xplus = xrange
        yminus, yplus = yrange

        return (
            (xs < self.NAXIS[1] + xplus)
            & (xs > xminus)
            & (ys < self.NAXIS[0] + yplus)
            & (ys > yminus)
        )

    def DISPL(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        return poly.POLY[self._polyname["DISPL"][order]](
            self._disp_data["DISPL"][order], x0, y0, t
        )

    def DDISPL(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        return poly.DPOLY[self._polyname["DISPL"][order]](
            self._disp_data["DISPL"][order], x0, y0, t
        )

    def DISPX(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        return -self.wx + poly.POLY[self._polyname["DISPX"][order]](
            self._disp_data["DISPX"][order], x0, y0, t
        )

    def DDISPX(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        return poly.DPOLY[self._polyname["DISPX"][order]](
            self._disp_data["DISPX"][order], x0, y0, t
        )

    def DISPY(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        return -self.wy + poly.POLY[self._polyname["DISPY"][order]](
            self._disp_data["DISPY"][order], x0, y0, t
        )

    def DDISPY(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
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
        return poly.DPOLY[self._polyname["DISPY"][order]](
            self._disp_data["DISPY"][order], x0, y0, t
        )

    def DISPXY(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        t: float | npt.NDArray[np.float64],
        theta: float = 0.0,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    def INVDISPL(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        wavelength: float | npt.NDArray[np.float64],
        t0: npt.NDArray[np.float64] = np.linspace(-1, 2, 40),
    ) -> npt.NDArray[np.float64]:
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
            return poly.INVPOLY[polyname](polydata, x0, y0, wavelength)
        except KeyError:
            invpolydata = self._disp_data["INVDISPL"][order]
            if invpolydata.size > 0:
                return poly.POLY[polyname](invpolydata, x0, y0, wavelength)
            else:
                xr = self.DISPL(order, x0, y0, t0)
                so = np.argsort(xr)
                interpolator = interp1d(
                    xr[so],
                    t0[so],
                    bounds_error=False,
                    fill_value="extrapolate",  # type: ignore
                )
                return interpolator(wavelength)

    def INVDISPX(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        dx: float | npt.NDArray[np.float64],
        t0: npt.NDArray[np.float64] = np.linspace(-1, 2, 40),
    ) -> npt.NDArray[np.float64]:
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
            return poly.INVPOLY[polyname](polydata, x0, y0, dx + self.wx)
        except KeyError:
            invpolydata = self._disp_data["INVDISPX"][order]
            if invpolydata.size > 0:
                return poly.POLY[polyname](invpolydata, x0, y0, dx)
            else:
                xr = self.DISPX(order, x0, y0, t0)
                so = np.argsort(xr)
                interpolator = interp1d(
                    xr[so],
                    t0[so],
                )
                return interpolator(dx)

    def INVDISPY(
        self,
        order: str,
        x0: float | npt.NDArray[np.float64],
        y0: float | npt.NDArray[np.float64],
        dy: float | npt.NDArray[np.float64],
        t0: npt.NDArray[np.float64] = np.linspace(-1, 2, 40),
    ) -> npt.NDArray[np.float64]:
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
            return poly.INVPOLY[polyname](polydata, x0, y0, dy + self.wy)
        except KeyError:
            invpolydata = self._disp_data["INVDISPY"][order]
            if invpolydata.size > 0:
                return poly.POLY[polyname](invpolydata, x0, y0, dy)
            else:
                yr = self.DISPY(order, x0, y0, t0)
                so = np.argsort(yr)
                interpolator = interp1d(
                    yr[so],
                    t0[so],
                )
                return interpolator(dy)
