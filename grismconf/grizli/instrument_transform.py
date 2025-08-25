"""
Transform instrument parameters to map orientation of intrinsic observation coordinates.

This class is a re-write of [Grizli's
class](https://github.com/gbrammer/grizli/blob/8e0ae2f89e162ab64fdc6f0b68ffa953f034e266/grizli/grismconf.py#L890)
to avoid hard coded values and convoluted decision tree.  

In this version, instrumental parameters are taken from json information
(which currently hard coded in `PARAMS`, but can be made dynamic) and provided as a dictionary.
"""
import json
from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt

# Parameters per recognized instrument
# Deals with aliases and default values
PARAMS = json.loads("""
{
    "*": {
        "facility": "default", 
        "module": "unknown",
        "instrument": "unknown",
        "rotation": 0.0, 
        "axis": "+x",
        "array_center": [ 507.5, 507.5 ] 
    },
    "NIRCAM": {
        "facility": "HST", 
        "array_center": [ 507.5, 507.5 ],
        "A": {
            "aliases": { "R": "GRISMR", "C": "GRISMC" },
            "GRISMR": { "rotation": 0.0, "axis": "+x" },
            "GRISMC": { "rotation": 90.0, "axis": "+y" }
        },
        "B": { 
            "aliases": { "R": "GRISMR", "C": "GRISMC" },
            "GRISMR": { "rotation": 180.0, "axis": "-x" },
            "GRISMC": { "rotation": 90.0, "axis": "+y" }
        }
    },
    "NIRISS": {
        "facility": "JWST", 
        "array_center": [ 1024.5, 1024.5 ],
        "*": {
            "aliases" : { "R": "GR150R", "C": "GR150C" },
            "GR150R": { "rotation": 270.0, "axis": "-y" },
            "GR150C": { "rotation": 180.0, "axis": "-x" }
        }
    }
}
""")


class InstrumentTransformation(dict):
    """ Transform instrument parameters to map orientation of intrinsic observation coordinates. 

    This class is a re-write of [Grizli's
    class](https://github.com/gbrammer/grizli/blob/8e0ae2f89e162ab64fdc6f0b68ffa953f034e266/grizli/grismconf.py#L890)
    to avoid hard coded values and convoluted decision tree.  
    
    In this version, instrumental parameters are taken from json information
    (which currently hard coded in `PARAMS`, but can be made dynamic) and provided as a dictionary.

    example usage:

    >>> InstrumentTransformation("NIRCAM", "A", "R")  # or InstrumentTransformation("NIRCAM", "A", "GRISMR")
     {'facility': 'HST',
      'module': 'A',
      'instrument': 'NIRCAM',
      'rotation': 0.0,
      'axis': '+x',
      'array_center': [507.5, 507.5],
      'grism': 'GRISMR'}
    >>> InstrumentTransformation("NIRISS", "*", "R") # or InstrumentTransformation("NIRISS", "*", "GR150R")
     {'facility': 'JWST',
      'module': '*',
      'instrument': 'NIRISS',
      'rotation': 270.0,
      'axis': '-y',
      'array_center': [1024.5, 1024.5],
      'grism': 'GR150R'}
    Note: when module does not matter, "*" or anything else can be used. If an instrument is not in the parameters
    a default transformation is provided.
    """
    def __init__(self, instrument: str, module: str, grism: str):
        super().__init__(self.set_params(instrument, module, grism))

    @classmethod
    def set_params(cls, instrument: str, module: str, grism: str) -> Dict[str, Any]:
        """ Set instrument parameters based on the provided instrument, module, and grism. """
        info = PARAMS["*"].copy()
        instname, inst_info = cls.resolve_module_name(PARAMS, instrument)

        # If default information return
        if instname == "*":
            return info

        info.update(
            {
                "facility": inst_info.get("facility", None),
                "instrument": instname,
                "array_center": inst_info.get("array_center", None),
            }
        )

        module_name, module_info = cls.resolve_module_name(inst_info, module)
        info["module"] = module_name
        grism_name, grism_info = cls.resolve_module_name(module_info, grism)
        info["grism"] = grism_name
        info.update(grism_info)

        return info

    @staticmethod
    def resolve_module_name(
        inst_info: Dict[str, Any], module: str
    ) -> Tuple[str, Dict[str, Any]]:
        """ Resolve the module name and its associated parameters. 

        Recognizes aliases and * as default values

        parameters
        ----------
        inst_info: Dict[str, Any]
            Instrument information dictionary.
        module: str
            Module name to resolve.
        
        Returns
        -------
        Tuple[str, Dict[str, Any]]
            Resolved module name (alias-resolved) and its associated parameters.
        """
        aliases = inst_info.get("aliases", {})
        if module in inst_info:
            return module, inst_info[module]
        elif module in aliases:
            return aliases[module], inst_info[aliases[module]]
        elif "*" in inst_info:
            return "*", inst_info["*"]
        raise LookupError(f"Module {module} not found in {inst_info}")

    def apply_rotation(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        reverse: bool = False,
    ) -> npt.NDArray[np.float64]:
        """
        Forward transform detector to +x.

        Rotate NIRISS and NIRCam coordinates such that slitless dispersion has
        wavelength increasing towards +x.

        Parameters
        ----------
        x, y : float or array-like
            Original detector coordinates

        Returns
        -------
        x, y : array-like
            Coordinates in rotated frame

        """
        theta = self.get("rotation", 0.0) / 180 * np.pi
        center = self.get("array_center", np.array([0.0, 0.0]))

        if reverse:
            theta = -theta

        rotmat = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )

        z = np.array([x, y]).T

        return ((z - center) @ rotmat + center).T