# Note: Code provided by R. Ryan.

import os
from typing import Union

import numpy as np
import requests
from astropy.io import fits
from astropy.modeling import polynomial
from astropy.table import Table
from jwst import datamodels


def fetch_reffile(filename: str, overwrite: bool = True, show: bool = True) -> None:
    """Function to fetch a reference file from the CRDS archive."""
    crdsurl = f"https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{filename}"

    if (not os.path.exists(filename)) or overwrite:
        if show:
            print(f"Fetching the file {filename}")
        r = requests.get(
            crdsurl, stream=True
        )  # params=params, headers=headers, stream=True)
        r.raise_for_status()
        with open(filename, "wb") as fobj:
            for chunk in r.iter_content(chunk_size=1_024_000):
                fobj.write(chunk)
    else:
        if show:
            print(f"Using local copy of {filename}")


def reformat_poly(obj: Union[polynomial.Polynomial1D, polynomial.Polynomial2D]) -> list:
    """
    Function to transform an astropy.Polynomial object into an list() that matches a given row of a grismconf file
    This function removes the astropy dependency after reading the configuration file.

    Parameters
    ----------
    obj : polynomial.Polynomial1D or polynomial.Polynomial2D
        The polynomial object to be reformatted.
    Returns
    -------
    list
        A list of coefficients corresponding to the polynomial object.
    Raises
    ------
    NotImplementedError
        If the polynomial object is not of type Polynomial1D or Polynomial2D.
    """

    coefs = list(np.zeros(len(obj.parameters), dtype=float))
    n = len(coefs)

    if isinstance(obj, polynomial.Polynomial1D):
        for i in range(n):
            coefs[i] = [getattr(obj, f"c{i}").value]
    elif isinstance(obj, polynomial.Polynomial2D):
        m = int(np.sqrt(8 * n + 1) - 1) // 2
        i = 0
        for j in range(m):
            for k in range(j + 1):
                coefs[i] = getattr(obj, f"c{j - k}_{k}").value
                i += 1
    else:
        raise NotImplementedError(
            f"Unsupported polynomial type: {type(obj)}. Expected Polynomial1D or Polynomial2D."
        )

    return coefs


def plot_sensitivity_curve(wavelength, sensitivity):
    """Display result from get_sensitivity(..., show=True) function.

    Parameters
    ----------
    wavelength : np.ndarray
        The wavelength array in microns.
    sensitivity : np.ndarray
        The sensitivity values in units of erg/s/cm^2/A per DN/s.
    """
    import matplotlib.pyplot as plt

    plt.plot(wavelength, sensitivity)
    plt.xlabel(r"Wavelength ($\mu m$)")
    plt.ylabel(r"DN/s per erg/s/cm^2/$\AA$")
    plt.grid()


def get_sensitivity(
    wfss_file: str, order: int = 1, show: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Fetch and process the sensitivity file for this observation.

    This function cleans up the content of the calibration file and changes the
    units of the sensitivity to be in flam per DN/s.

    Parameters
    ----------
    wfss_file : str
        The name of the WFSS file to process.
    order : int, optional
        The order of the grism to use for the sensitivity calculation, by default 1.
    show : bool, optional
        If True, the sensitivity curve will be plotted, by default False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the wavelength array and the sensitivity values in units of erg/s/cm^2/A per DN/s.

    Raises
    ------
    NameError
        If the WFSS file does not contain the required WCS information.
    ValueError
        If the sensitivity file does not contain the expected filter, pupil, and order information.
    """

    # We need to get the pixel size of the detector.
    # We also get the PUPIL and FILTER name

    # FIXME: Direct approach without datamodel -- general solution needed
    # with fits.open(wfss_file) as fin:
    #    pixel_area = fin[1].header["PIXAR_SR"]  # type: ignore[no-untyped-call]
    #    pupil = fin[0].header["PUPIL"]  # type: ignore[no-untyped-call]
    #    filter = fin[0].header["FILTER"]  # type: ignore[no-untyped-call]

    with datamodels.open(wfss_file) as dm:
        # parameters = dm.get_crds_parameters()
        sensitivity_file = dm.meta.ref_file.photom.name[7:]
        pupil = dm.meta.instrument.pupil
        filter = dm.meta.instrument.filter

    fetch_reffile(sensitivity_file, overwrite=False, show=False)

    tab = Table.read(sensitivity_file)
    pixel_area = tab.meta["PIXAR_SR"]
    ok = (tab["filter"] == filter) & (tab["pupil"] == pupil) & (tab["order"] == order)
    w = np.asarray(tab[ok][0]["wavelength"])
    s = np.asarray(tab[ok][0]["relresponse"])
    photmjsr = tab[ok][0]["photmjsr"]
    ok = np.nonzero(w)
    w = w[ok]
    s = s[ok]

    # The sensitivity is by default in units of Mjy per SR per DN/s (per pixel) which we convert to
    # the more traditional value of erg/s/cm^2/A per DN/s
    c = 29_979_245_800.0
    s2 = (w * 1e4) / c * (w / 1e8) / (s * photmjsr * 1e6 * 1e-23 * pixel_area) * 10000

    if show:
        plot_sensitivity_curve(w, s2)

    return w, s2


def specwcs_poly(wfss_file, order=1):
    """Get the polynomial WCS for the specified WFSS file.
    Parameters
    ----------
    wfss_file : str
        The name of the WFSS file to process.
    order : int, optional
        The order of the grism to use for the polynomial WCS, by default 1.

    Returns
    -------
    tuple[dict, dict, dict, dict]
        A tuple containing dictionaries for the x, y, and l models, and the sensitivity data.

    Raises
    ------
    NameError
        If the WFSS file does not contain the required WCS information.
    """
    _DISPX_data = {}
    _DISPY_data = {}
    _DISPL_data = {}
    SENS_data = {}

    with datamodels.open(wfss_file) as dm:
        t = dm.meta.wcs.get_transform(
            from_frame="detector",
            to_frame="grism_detector",
        )[-1]
        for order, xmodel, ymodel, lmodel in zip(
            t.orders, t.xmodels, t.ymodels, t.lmodels
        ):
            sorder = f"{order:+}"
            _DISPX_data[sorder] = np.array([reformat_poly(p2d) for p2d in xmodel])
            if len(xmodel) == 1:
                _DISPX_data[sorder] = _DISPX_data[sorder][0]

            _DISPY_data[sorder] = np.array([reformat_poly(p2d) for p2d in ymodel])
            if len(ymodel) == 1:
                _DISPY_data[sorder] = _DISPY_data[sorder][0]

            # The lmodels are (5,) for the 5 orders, not (5, 3) for NIRISS
            try:
                _DISPL_data[sorder] = np.array([reformat_poly(p2d) for p2d in lmodel])
            except TypeError:
                _DISPL_data[sorder] = np.array(reformat_poly(lmodel))[0]

            SENS_data[sorder] = get_sensitivity(wfss_file, order=order)

    return _DISPX_data, _DISPY_data, _DISPL_data, SENS_data
