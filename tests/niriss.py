import os
import tarfile
from glob import glob
from pathlib import Path
from typing import List, Mapping, Tuple

import numpy as np
import numpy.typing as npt
import requests

from grismconf import grismconf2
from grismconf.grismconf2 import GrismconfParser, transform_v2Conf


def fetch_file(url: str, overwrite: bool = False) -> str:
    """Function to fetch an example configuration file."""
    filename = url.split("/")[-1].split("?")[0]
    if (not os.path.exists(filename)) or overwrite:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename, "wb") as fobj:
            for chunk in r.iter_content(chunk_size=1_024_000):
                fobj.write(chunk)
    return filename


def fix_test_data(source: str = "niriss_test_data"):
    """ The zenodo files are probably from various iterations and are not consistent.
    This fixes the path to sensitivity data
    """
    config_files = glob(f"{source}/*.conf")
    for conf_file in config_files:
        with open(conf_file, "r") as f:
            content = f.readlines()

        # Fix SENSITIVITY lines
        fixed_content = []
        for line in content:
            if line.startswith("#"):
                fixed_content.append(line)
                continue
            if "SENSITIVITY_" in line:
                parts = line.split()
                if len(parts) > 1 and parts[1].endswith(".fits"):
                    parts[1] = parts[1].replace("wfss-grism-configuration/", "")
                    parts[1] = ".".join(parts[1].split('.')[:3] + ["wd1657.p1.sens.fits"])
                fixed_content.append(" ".join(parts) + "\n")
            else:
                fixed_content.append(line)

        with open(conf_file, "w") as f:
            f.writelines(fixed_content)


def download_test_data() -> List[str]:
    """Download test data files."""
    urls = [
        "https://zenodo.org/records/7628094/files/niriss_config_221215.tar.gz?download=1",
        "https://zenodo.org/records/7628094/files/niriss_sens_221215.tar.gz?download=1",
    ]

    # only keep non-hidden files with extension .conf and .fits
    def extraction_filter(tarinfo: tarfile.TarInfo, *args) -> tarfile.TarInfo | None:
        basename, ext = os.path.splitext(tarinfo.name)
        if ext in [".fits", ".conf"] and not basename.startswith("."):
            return tarinfo

    destination = "niriss_test_data"
    os.makedirs(destination, exist_ok=True)
    downloaded_files = [destination]
    for url in urls:
        fname = fetch_file(url, overwrite=True)
        downloaded_files.append(fname)
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(path=destination, filter=extraction_filter)
    fix_test_data(destination)
    return downloaded_files


def test_transform_v2conf(fname="niriss_test_data/GR150C.F150W.221215.conf"):
    """test the conversion from v2.0 to v1.0 conventions"""
    c = GrismconfParser.from_file(fname)
    C = transform_v2Conf(c, quiet=True)

    assert len(c) == len(C), "Transformed configuration does not match original length"

    test_keys = [
        ("DYDX_A_0", "DYDX_+1_0"),
        ("DYDX_B_0", "DYDX_0_0"),
        ("XOFF_A", "XOFF_+1"),
        ("DLDP_A_0", "DLDP_+1_0"),
        ("DLDP_B_0", "DLDP_0_0"),
    ]
    for key2, key1 in test_keys:
        # test arrays and values
        try:
            for val1, val2 in zip(C[key1], c[key2]):
                assert val1 == val2, (
                    f"Transformed value {key1}:{val1} does not match original value {key2}:{val2}."
                )
        except TypeError:
            assert C[key1] == c[key2], (
                f"Transformed key {key1} does not match original key {key2}"
            )


class GrismConf(grismconf2.GrismConf):
    def __init__(self, filename: str, dirfilter: str | None = None):
        """Initialize GrismConf object from a grismconf file."""

        # Internals
        self.source: Path = Path(filename)
        self.config: GrismconfParser = transform_v2Conf(
            GrismconfParser.from_file(filename)
        )

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


if __name__ == "__main__":
    fname = "wfss-grism-configuration/GR150C.F150W.221215.conf"
    C = grismconf2.transform_v2Conf(grismconf2.GrismconfParser.from_file(fname))
