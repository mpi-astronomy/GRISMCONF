import os
import re
from typing import List

import numpy as np
import pytest
import requests

import grismconf

EXAMPLE_NIRCAM_CONF_URL = "https://raw.githubusercontent.com/npirzkal/GRISM_NIRCAM/refs/heads/master/V9/NIRCAM_F250M_modA_C.conf"
EXAMPLE_WFC3_CONF_URL = "https://raw.githubusercontent.com/npirzkal/GRISM_WFC3/refs/heads/master/IR/G102.conf"


def fetch_file(url: str, overwrite: bool = True) -> str:
    """Function to fetch an example configuration file."""
    filename = url.split("/")[-1]
    if (not os.path.exists(filename)) or overwrite:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename, "wb") as fobj:
            for chunk in r.iter_content(chunk_size=1_024_000):
                fobj.write(chunk)
    return filename


def fetch_conf_dependencies(
    url: str,
    conf_fname: str,
    overwrite: bool = True,
    pattern: str = r".*\s(.*fits)$",
    show: bool = False,
) -> List[str]:
    """Identify and fetch all dependencies of a configuration file.
    This function reads a configuration file and fetches all files that match the given pattern.
    The pattern is a regular expression that should match the file names in the configuration file.
    The default pattern matches lines ending with a .fits file name.
    """

    search = re.compile(pattern)
    URL_base = url.rsplit("/", 1)[0] + "/{fname}"

    file_list = []
    with open(conf_fname, "r") as f:
        for line in f:
            if match := search.match(line):
                fname = match.group(1)
                if show:
                    print(
                        f"{line.strip()} \n Fetching {fname} from {URL_base.format(fname=fname)}"
                    )
                file_list.append(
                    fetch_file(URL_base.format(fname=fname), overwrite=overwrite)
                )
    return file_list


@pytest.fixture(scope="session")
def collect_test_data():
    """Collects test data for the GrismConf class."""
    fname = fetch_file(EXAMPLE_NIRCAM_CONF_URL)
    temp_files = [fname]

    # Fetch dependencies
    temp_files += fetch_conf_dependencies(EXAMPLE_NIRCAM_CONF_URL, fname)

    yield fname, temp_files

    # Clean up temporary files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_example1(collect_test_data):
    fname, _ = collect_test_data

    # example position
    x0, y0 = 500.5, 600.1

    # Load the Grism Configuration file
    C = grismconf.Config(fname)

    # edges of spectra (use t=0 and t=1)
    dx01 = C.DISPX("+1", x0, y0, np.array([0, 1]))
    assert dx01.shape == (2,)
    assert dx01[0] < dx01[1], "dx01 should be increasing"
    assert np.isclose(dx01[0], -49.85361111, atol=1e-6), (
        "dx01[0] should be close to -49.85361111"
    )
    assert np.isclose(dx01[1], -48.48791052, atol=1e-6), (
        "dx01[1] should be close to  -48.48791052"
    )

    # Get a list of all dxs value along this trace
    dxs = np.arange(dx01[0], dx01[1])
    # Compute the t values corresponding to the exact offsets dxs
    ts = C.INVDISPX("+1", x0, y0, dxs)
    assert ts.shape == (len(dxs),), "ts should have the same length as dxs"
    assert np.all(ts >= 0) and np.all(ts <= 1), "ts should be in the range [0, 1]"
    assert np.isclose(ts[0], 0.0, atol=1e-6), "ts[0] should be close to 0.0"
    assert np.isclose(ts[-1], 0.72035807, atol=1e-6), (
        "ts[-1] should be close to 0.72035807"
    )

    # dys: [-1530.876494   -1373.38785167], wavs: [2.49696493 2.64881764]
    # Compute the dys values for the same pixels
    dys = C.DISPY("+1", x0, y0, ts)
    assert dys.shape == (len(dxs),), "dys should have the same length as dxs"
    assert np.all(dys <= 0), "dys should be non-positive"
    assert np.isclose(dys[0], -1530.876494, atol=1e-6), (
        "dys[0] should be close to -1530.876494"
    )
    assert np.isclose(dys[-1], -1373.38785167, atol=1e-6), (
        "dys[-1] should be close to -1373.38785167"
    )

    # Compute wavelength of each of the pixels
    wavs = C.DISPL("+1", x0, y0, ts)
    assert wavs.shape == (len(dxs),), "wavs should have the same length as dxs"
    assert np.all(wavs >= 2.4) and np.all(wavs <= 2.7), (
        "wavs should be in the range [2.4, 2.7]"
    )
    assert np.isclose(wavs[0], 2.49696493, atol=1e-6), (
        "wavs[0] should be close to 2.49696493"
    )
    assert np.isclose(wavs[-1], 2.64881764, atol=1e-6), (
        "wavs[-1] should be close to 2.64881764"
    )

    # Clean up temporary files
    # FIXME: Uncomment the following lines if you want to remove the temporary files after the test
    # for temp_file in temp_files:
    #     if os.path.exists(temp_file):
    #         os.remove(temp_file)


def test_example2(collect_test_data):
    fname, _ = collect_test_data

    # example position
    x0, y0 = 500.5, 600.1

    # Load the Grism Configuration file
    C = grismconf.Config(fname)

    ratio_12 = C.DDISPL("+1", x0, y0, 0) / C.DDISPX("+2", x0, y0, 0)

    assert np.isclose(ratio_12, -0.523209215269946, atol=1e-14), (
        "ratio_12 should be close to -0.523209215269946"
    )
