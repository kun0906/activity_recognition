"""Testing for feature A
"""
import pytest as pytest

from models.classical.detector_feature_A import get_shape


# @pytest.mark.parametrize({"X":[50, 100, 150, 500], "Y": [10, 200], 'Z': []})
@pytest.mark.parametrize("X", [[10], [20]])
@pytest.mark.parametrize("Y", [[10]])
def test_get_shape(X, Y, Z=[]):
	# Test get_X_y()
	X, Y, Z = get_shape(X, Y, Z)
	print(X, Y, Z)
	assert X == Y
