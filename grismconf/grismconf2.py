import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple

import numpy as np
import numpy.typing as npt
from astropy.io import fits
from scipy.interpolate import interp1d

# from . import poly, specwcs
from grismconf import poly


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
        "FWCPOS_REF": float,
        # NIRCAM FIELDS
        "NAXIS": int,
        "DISPL": float,
        "DISPY": float,
        "DISPX": float,
        "XRANGE": float,
        "YRANGE": float,
        "POMX": float,
        "POMY": float,
        "WEDGE": float,
        # NIRISS FIELDS
        "BEAM": int,
        "XOFF": float,
        "YOFF": float,
        "MMAG_EXTRACT": int,
        "DYDX_ORDER": int,
        "DISP_ORDER": int,
        "DLDP": float,
        "DYDX": float,
    }

    def __getitem__(self, key: Any) -> Any:
        """Get the value(s) associated with a key or pattern in the configuration."""
        try:
            return super().__getitem__(key)
        except KeyError:
            # not direct match, try pattern search
            matcher = re.compile(rf"{key}")
            matches = self.__class__(
                {k: v for k, v in self.items() if matcher.match(k)}
            )
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
            what = elements[0].split("_")
            value_mapper = lambda x: x  # noqa: E731, default identity
            for k in reversed(range(len(what))):
                key = "_".join(what[:k])
                if key in mapper:
                    value_mapper = mapper[key]
                    break
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


def transform_v2Conf(c: GrismconfParser, quiet: bool = False) -> GrismconfParser:
    """Transform a v2.0 configuration to a v1.0 configuration.

    This function maps the v2.0 order names (A, B, C, ...) to their corresponding
    v1.0 order values (+1, 0, +2, ...).

    The mapping between the two is hard coded and taken from [Grizly](https://github.com/gbrammer/grizli)

    The detection of v2 scheme is based of the presence of keys like "BEAM[A|B|C|D|E|F]" in the configuration.
    """
    v2_order_names = {
        "A": "+1",
        "B": "0",
        "C": "+2",
        "D": "+3",
        "E": "-1",
        "F": "+4",
    }

    # detect format of the configuration
    try:
        assert len(c["BEAM[" + "|".join(v2_order_names) + "]"]) > 0, (
            "impossible v2 detection error"
        )
        if not quiet:
            print("Detected configuration format: v2.0 (orders defined as A, B, C...)")
    except KeyError:
        if not quiet:
            print("Detected configuration format: v1.0 (orders defined as +<order>)")
        return c

    # copy over all values not associated with the beam info
    new_conf = GrismconfParser()
    beam_matcher_expr = r".*_{order_name}_\d+|.*_{order_name}$|BEAM{order_name}$"
    order_name = "[" + "".join(v2_order_names) + "]"
    copy_keys = [
        k
        for k in c.keys()
        if not re.compile(beam_matcher_expr.format(order_name=order_name)).match(k)
    ]
    new_conf.update({k: c[k] for k in copy_keys})

    # copy beam info and rename
    which_beams = [(k, v) for k, v in v2_order_names.items() if f"BEAM{k}" in c]
    for order_name, order_value in which_beams:
        try:
            subdata = c[beam_matcher_expr.format(order_name=order_name)]
        except KeyError:
            subdata = {}
        for key, value in subdata.items():
            if key.startswith("BEAM"):
                new_key = f"{key[:-1]}_{order_value}"
                new_conf[new_key] = [int(k) for k in value]
                continue
            new_key = key.replace(f"_{order_name}", f"_{order_value}")
            new_conf[new_key] = value
    return new_conf


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
        self._disp_data: Mapping[str, Mapping[str, npt.NDArray[np.float64]]] = (
            self._get_disp_data()
        )

        # shapes of the DISP and INVDISP coefficients per order
        # missing orders will be defined as shape (0, 0)
        self._polyname: dict[str, dict[str, tuple[int, int]]] = {}
        for key in self._disp_data.keys():
            self._polyname[key] = {}
            for order in self.orders:
                self._polyname[key][order] = np.shape(self._disp_data[key][order])

        # get sensitivity data
        self._sens_data: Mapping[
            str, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ] = self._get_sens_data()

        # set wavelength range for each order
        self.WRANGE: Mapping[str, Tuple[float, float]] = self._set_wrange(
            self._sens_data
        )

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

    def _get_disp_data(self) -> Mapping[str, Mapping[str, npt.NDArray[np.float64]]]:
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

    def _get_sens_data(
        self,
    ) -> Mapping[str, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Extracts the sensitivity data from the configuration."""
        # get sensitivity data
        data = {}
        for order in self.orders:
            data[order] = self._get_sensitivity(order)
        return data

    @staticmethod
    def _set_wrange(
        sens_data: Mapping[
            str, Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ],
    ) -> Mapping[str, Tuple[float, float]]:
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

    def _get_sensitivity(
        self, order: str
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Helper function that looks for the name of the sensitivity file,
        reads it and stores the content in a simple list
        [WAVELENGTH, SENSITIVITY].
        """
        fname = self._get_file_path(f"SENSITIVITY_{order}")

        with fits.open(fname) as fin:
            wavs = fin[1].data.field("WAVELENGTH")[:] * 1  # pyright: ignore[reportAttributeAccessIssue]
            sens = fin[1].data.field("SENSITIVITY")[:] * 1  # pyright: ignore[reportAttributeAccessIssue]

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
    def POM_POLYGON(self) -> npt.NDArray[np.float64]:
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
    def _DISPL_data(self) -> Mapping:
        return self._disp_data["DISPL"]

    @property
    def _DISPX_data(self) -> Mapping:
        return self._disp_data["DISPX"]

    @property
    def _DISPY_data(self) -> Mapping:
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
