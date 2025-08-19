import pytest
import numpy as np
import numpy.testing as npt

from grisconf.instrument_transform import InstrumentTransformation, PARAMS


class TestInstrumentTransformation:
    """Test suite for InstrumentTransformation class."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        # Test NIRCAM A GRISMR
        transform = InstrumentTransformation("NIRCAM", "A", "GRISMR")
        assert transform["instrument"] == "NIRCAM"
        assert transform["module"] == "A"
        assert transform["grism"] == "GRISMR"
        assert transform["facility"] == "HST"
        assert transform["rotation"] == 0.0
        assert transform["axis"] == "+x"
        assert transform["array_center"] == [507.5, 507.5]

    def test_init_with_aliases(self):
        """Test initialization using aliases."""
        # Test NIRCAM A R (alias for GRISMR)
        transform = InstrumentTransformation("NIRCAM", "A", "R")
        assert transform["grism"] == "GRISMR"
        assert transform["rotation"] == 0.0
        assert transform["axis"] == "+x"

        # Test NIRCAM A C (alias for GRISMC)
        transform = InstrumentTransformation("NIRCAM", "A", "C")
        assert transform["grism"] == "GRISMC"
        assert transform["rotation"] == 90.0
        assert transform["axis"] == "+y"

    def test_niriss_configuration(self):
        """Test NIRISS specific configuration."""
        # Test NIRISS GR150R
        transform = InstrumentTransformation("NIRISS", "*", "GR150R")
        assert transform["instrument"] == "NIRISS"
        assert transform["facility"] == "JWST"
        assert transform["rotation"] == 270.0
        assert transform["axis"] == "-y"
        assert transform["array_center"] == [1024.5, 1024.5]
        assert transform["grism"] == "GR150R"

        # Test NIRISS with alias
        transform = InstrumentTransformation("NIRISS", "*", "R")
        assert transform["grism"] == "GR150R"

    def test_nircam_modules(self):
        """Test different NIRCAM modules (A and B)."""
        # Module A
        transform_a = InstrumentTransformation("NIRCAM", "A", "GRISMR")
        assert transform_a["module"] == "A"
        assert transform_a["rotation"] == 0.0

        # Module B
        transform_b = InstrumentTransformation("NIRCAM", "B", "GRISMR")
        assert transform_b["module"] == "B"
        assert transform_b["rotation"] == 180.0
        assert transform_b["axis"] == "-x"

    def test_unknown_instrument_defaults(self):
        """Test behavior with unknown instrument (should use defaults)."""
        transform = InstrumentTransformation("UNKNOWN", "MODULE", "GRISM")
        # Should fall back to defaults from "*"
        assert transform["facility"] == "default"
        assert transform["rotation"] == 0.0
        assert transform["axis"] == "+x"
        assert transform["array_center"] == [507.5, 507.5]

    def test_resolve_module_name_method(self):
        """Test the resolve_module_name static method."""
        # Test direct module name
        module_name, module_info = InstrumentTransformation.resolve_module_name(
            PARAMS["NIRCAM"]["A"], "GRISMR"
        )
        assert module_name == "GRISMR"
        assert module_info["rotation"] == 0.0

        # Test alias resolution
        module_name, module_info = InstrumentTransformation.resolve_module_name(
            PARAMS["NIRCAM"]["A"], "R"
        )
        assert module_name == "GRISMR"
        assert module_info["rotation"] == 0.0

        # Test wildcard fallback
        module_name, module_info = InstrumentTransformation.resolve_module_name(
            PARAMS["NIRISS"], "anything"
        )
        assert module_name == "*"

    def test_resolve_module_name_error(self):
        """Test that resolve_module_name raises LookupError for invalid modules."""
        test_info = {"valid_module": {}, "aliases": {"alias": "valid_module"}}
        
        with pytest.raises(LookupError, match="Module invalid not found"):
            InstrumentTransformation.resolve_module_name(test_info, "invalid")

    def test_apply_rotation_identity(self):
        """Test apply_rotation with zero rotation (identity transformation)."""
        transform = InstrumentTransformation("NIRCAM", "A", "GRISMR")  # rotation = 0.0
        
        x, y = 100.0, 200.0
        x_rot, y_rot = transform.apply_rotation(x, y)
        
        npt.assert_allclose([x_rot, y_rot], [x, y], rtol=1e-10)

    def test_apply_rotation_90_degrees(self):
        """Test apply_rotation with 90-degree rotation."""
        transform = InstrumentTransformation("NIRCAM", "A", "GRISMC")  # rotation = 90.0
        
        # Test single point
        x, y = 100.0, 200.0
        x_rot, y_rot = transform.apply_rotation(x, y)
        
        # For 90-degree rotation around center [507.5, 507.5]:
        # Expected result can be calculated manually
        center = np.array([507.5, 507.5])
        theta = 90.0 / 180 * np.pi
        rotmat = np.array([[np.cos(theta), -np.sin(theta)], 
                          [np.sin(theta), np.cos(theta)]])
        expected = ((np.array([x, y]) - center) @ rotmat + center)
        
        npt.assert_allclose([x_rot, y_rot], expected, rtol=1e-10)

    def test_apply_rotation_array_input(self):
        """Test apply_rotation with array inputs."""
        transform = InstrumentTransformation("NIRISS", "*", "GR150R")  # rotation = 270.0
        
        x = np.array([100.0, 200.0, 300.0])
        y = np.array([150.0, 250.0, 350.0])
        
        x_rot, y_rot = transform.apply_rotation(x, y)
        
        # Should return arrays of same shape
        assert x_rot.shape == x.shape
        assert y_rot.shape == y.shape
        assert isinstance(x_rot, np.ndarray)
        assert isinstance(y_rot, np.ndarray)

    def test_apply_rotation_reverse(self):
        """Test apply_rotation with reverse=True."""
        transform = InstrumentTransformation("NIRCAM", "B", "GRISMR")  # rotation = 180.0
        
        x, y = 100.0, 200.0
        
        # Forward and reverse should be inverses
        x_rot, y_rot = transform.apply_rotation(x, y)
        x_back, y_back = transform.apply_rotation(x_rot, y_rot, reverse=True)
        
        npt.assert_allclose([x_back, y_back], [x, y], rtol=1e-10)

    def test_apply_rotation_different_centers(self):
        """Test apply_rotation with different array centers."""
        # NIRISS has different center than NIRCAM
        transform_niriss = InstrumentTransformation("NIRISS", "*", "GR150R")
        transform_nircam = InstrumentTransformation("NIRCAM", "A", "GRISMR")
        
        x, y = 1000.0, 1000.0
        
        # Same rotation angle but different centers should give different results
        # (if they had the same rotation angle, which they don't in this case)
        x_niriss, y_niriss = transform_niriss.apply_rotation(x, y)
        x_nircam, y_nircam = transform_nircam.apply_rotation(x, y)
        
        # Just verify they return valid arrays
        assert isinstance(x_niriss, (float, np.ndarray))
        assert isinstance(y_niriss, (float, np.ndarray))
        assert isinstance(x_nircam, (float, np.ndarray))
        assert isinstance(y_nircam, (float, np.ndarray))

    def test_dictionary_interface(self):
        """Test that the class behaves as a dictionary."""
        transform = InstrumentTransformation("NIRCAM", "A", "GRISMR")
        
        # Test dictionary access
        assert "instrument" in transform
        assert transform.get("rotation") == 0.0
        assert transform.get("nonexistent", "default") == "default"
        
        # Test modification
        transform["custom_param"] = "test_value"
        assert transform["custom_param"] == "test_value"

    def test_comprehensive_parameter_combinations(self):
        """Test various parameter combinations for comprehensive coverage."""
        test_cases = [
            ("NIRCAM", "A", "R", "GRISMR", 0.0, "+x"),
            ("NIRCAM", "A", "C", "GRISMC", 90.0, "+y"),
            ("NIRCAM", "B", "R", "GRISMR", 180.0, "-x"),
            ("NIRCAM", "B", "C", "GRISMC", 90.0, "+y"),
            ("NIRISS", "*", "R", "GR150R", 270.0, "-y"),
            ("NIRISS", "*", "C", "GR150C", 180.0, "-x"),
            ("NIRISS", "any", "GR150R", "GR150R", 270.0, "-y"),
        ]
        
        for instrument, module, grism_input, expected_grism, expected_rotation, expected_axis in test_cases:
            transform = InstrumentTransformation(instrument, module, grism_input)
            assert transform["grism"] == expected_grism, f"Failed for {instrument}/{module}/{grism_input}"
            assert transform["rotation"] == expected_rotation, f"Failed rotation for {instrument}/{module}/{grism_input}"
            assert transform["axis"] == expected_axis, f"Failed axis for {instrument}/{module}/{grism_input}"

    def test_set_params_classmethod(self):
        """Test the set_params class method directly."""
        params = InstrumentTransformation.set_params("NIRCAM", "A", "GRISMR")
        
        assert isinstance(params, dict)
        assert params["instrument"] == "NIRCAM"
        assert params["module"] == "A"
        assert params["grism"] == "GRISMR"
        assert params["facility"] == "HST"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with empty strings (should fall back to defaults)
        transform = InstrumentTransformation("", "", "")
        assert transform["facility"] == "default"
        
        # Test rotation at array center (should be identity)
        transform = InstrumentTransformation("NIRCAM", "A", "GRISMC")  # 90 degree rotation
        center = transform["array_center"]
        x_rot, y_rot = transform.apply_rotation(center[0], center[1])
        npt.assert_allclose([x_rot, y_rot], center, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
