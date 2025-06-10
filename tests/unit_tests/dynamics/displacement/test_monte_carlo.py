# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from .displacement_settings import DisplacementSettings

    @staticmethod
    def test_calculate_displacement(backend_class):
        # Arrange
        settings = DisplacementSettings()
        value_a = 0.1
        value_b = 0.2
        weight = 0.125
        settings.courant_field_data = (
            np.array([[value_a, value_b]]).T,
            np.array([[0, 0]]),
        )
        settings.positions = [[weight], [0]]
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ExplicitInSpace", adaptive=False, enable_monte_carlo=True
        )
        # Act
        sut.calculate_displacement(
            displacement=sut.displacement,
            courant=sut.courant,
            cell_origin=particulator.attributes["cell origin"],
            position_in_cell=particulator.attributes["position in cell"],
            cell_id=particulator.attributes["cell id"],
        )

        # Assert
        if backend_class.__name__ == "ThrustRTC":
            np.testing.assert_equal(
                sut.displacement[0, slice(0, 1)].to_ndarray(),
                0.1125
            )

        if backend_class.__name__ == "Numba":
            np.testing.assert_equal(
                sut.displacement[0, slice(0, 1)].to_ndarray(),
                0.125
            )

    @staticmethod
    def test_calculate_displacement_2(backend_class):
        # Arrange
        settings = DisplacementSettings()
        value_a = 1
        value_b = 1
        weight = 0.125
        settings.courant_field_data = (
            np.array([[value_a, value_b]]).T,
            np.array([[0, 0]]),
        )
        settings.positions = [[weight], [0]]
        sut, particulator = settings.get_displacement(
            backend_class, scheme="ExplicitInSpace", adaptive=False, enable_monte_carlo=True
        )
        # Act
        sut.calculate_displacement(
            displacement=sut.displacement,
            courant=sut.courant,
            cell_origin=particulator.attributes["cell origin"],
            position_in_cell=particulator.attributes["position in cell"],
            cell_id=particulator.attributes["cell id"],
        )

        # Assert
        if backend_class.__name__ == "ThrustRTC":
            np.testing.assert_equal(
                sut.displacement[0, slice(0, 1)].to_ndarray(),
                1.
            )

        if backend_class.__name__ == "Numba":
            np.testing.assert_equal(
                sut.displacement[0, slice(0, 1)].to_ndarray(),
                2.125
            )