import napari
import numpy as np
import pytest

from ..utils import (
    align_value_length,
    points_to_ts_indices,
    shape_to_ts_indices,
    to_layer_space,
    to_world_space,
)

# fixtures
# make numpy random seed fixed
SEED = 123


@pytest.fixture(autouse=True)
def mock_random(monkeypatch: pytest.MonkeyPatch):
    def stable_random(*args, **kwargs):
        rs = np.random.RandomState(SEED)
        return rs.random(*args, **kwargs)

    def stable_randint(*args, **kwargs):
        rs = np.random.RandomState(SEED)
        return rs.randint(*args, **kwargs)

    def stable_uniform(*args, **kwargs):
        rs = np.random.RandomState(SEED)
        return rs.uniform(*args, **kwargs)

    monkeypatch.setattr("numpy.random.random", stable_random)
    monkeypatch.setattr("numpy.random.randint", stable_randint)
    monkeypatch.setattr("numpy.random.uniform", stable_uniform)


@pytest.fixture
def image_layer3d():
    data = np.random.random((100, 100, 100))
    layer = napari.layers.Image(data)
    yield layer


@pytest.fixture
def points():
    yield np.random.uniform(0, 115, (10, 4))


@pytest.fixture
def shape():
    arr = np.empty((4, 4), dtype=np.float16)
    arr[:, 0] = np.random.randint(0, 9)
    arr[:, 1] = np.random.randint(0, 99)
    nodes = sorted(np.random.uniform(0, 50, (4)))
    arr[0, 2:] = nodes[1], nodes[0]
    arr[1, 2:] = nodes[1], nodes[3]
    arr[2, 2:] = nodes[2], nodes[3]
    arr[3, 2:] = nodes[2], nodes[0]
    yield arr


# tests
def test_to_world_space_no_transformation(points, image_layer3d):
    transf = to_world_space(points[:, -3:], image_layer3d)
    assert isinstance(transf, np.ndarray)
    assert np.array_equal(points[:, -3:], transf)
    assert transf.dtype == np.float64


def test_to_world_space_with_data_empty(image_layer3d):
    coords = np.array([[]], dtype=np.int8)
    transf = to_world_space(coords, image_layer3d)
    assert np.array_equal(coords, transf)
    assert transf.dtype == np.int8


def test_to_world_space_transformation(points, image_layer3d):
    t = np.random.uniform(-10, 10, (3))
    image_layer3d.translate = t
    s = np.random.uniform(0.1, 2, (3))
    image_layer3d.scale = s
    target = np.dot(points[:, -3:], np.diag(s)) + t

    transf = to_world_space(points[:, -3:], image_layer3d)
    assert np.allclose(target, transf)


def test_to_world_space_dim_missmatch(points, image_layer3d):
    t = np.random.uniform(-10, 10, (3))
    image_layer3d.translate = t
    s = np.random.uniform(0.1, 2, (3))
    image_layer3d.scale = s
    with pytest.raises(ValueError, match="different dimensionality"):
        to_world_space(points, image_layer3d)
    with pytest.raises(ValueError, match="different dimensionality"):
        to_world_space(points[:, :2], image_layer3d)


def test_to_layer_space_no_transformation(points, image_layer3d):
    transf = to_layer_space(points[:, -3:], image_layer3d)
    assert isinstance(transf, np.ndarray)
    assert np.array_equal(points[:, -3:], transf)
    assert transf.dtype == np.float64


def test_to_layer_space_with_data_empty(image_layer3d):
    coords = np.array([[]], dtype=np.int8)

    transf = to_layer_space(coords, image_layer3d)
    assert np.array_equal(coords, transf)
    assert transf.dtype == np.int8


def test_to_layer_space_transformation(points, image_layer3d):
    t = np.random.uniform(-10, 10, (4))
    image_layer3d.translate = t[-3:]
    s = np.random.uniform(-2, 2, (4))
    image_layer3d.scale = s[-3:]
    target = np.dot(
        points[:, -3:] - t[-3:],
        np.linalg.inv(np.diag(s[-3:])),
    )

    # Test with higher dim points
    transf_higher = to_layer_space(points, image_layer3d)
    assert np.allclose(target, transf_higher)

    # Test with same dim points
    transf_same = to_layer_space(points[:, -3:], image_layer3d)
    assert np.allclose(target, transf_same)

    # Test with lower dim points
    transf_lower = to_layer_space(points[:, -2:], image_layer3d)
    assert np.allclose(target[:, -2:], transf_lower)


def test_points_to_ts_indices_empty_data(image_layer3d):
    points = np.array([])
    result = points_to_ts_indices(points, image_layer3d)
    assert result == []


def test_points_to_ts_indices_invalid_dim(image_layer3d):
    points = np.array([[1]])
    with pytest.raises(
        ValueError,
        match="Point coordinates can have only one dimension less than layer.",
    ):
        points_to_ts_indices(points, image_layer3d)


def test_points_to_ts_indices(points, image_layer3d, monkeypatch):
    # Test conversion of point coordinates to time series indices
    def mock_func(points, layer):
        return points

    monkeypatch.setattr(
        "napari_time_series_plotter.utils.to_layer_space", mock_func
    )
    target_no_slice_smaller_dim = np.round(
        [p for p in points[:, -2:] if max(p) < 100]
    )[:, -2:]
    target_no_slice = np.round(
        [p for p in points[:, -3:] if max(p[1:]) < 100]
    )[:, -2:]

    # Test with points of one dim smaller
    result = points_to_ts_indices(points[:, -2:], image_layer3d)
    assert all(isinstance(p[0], slice) for p in result)
    assert np.allclose(target_no_slice_smaller_dim, [p[1:] for p in result])

    # test with points same dim
    result = points_to_ts_indices(points[:, -3:], image_layer3d)
    assert all(isinstance(p[0], slice) for p in result)
    assert np.allclose(target_no_slice, [p[1:] for p in result])

    # test with points higher dim
    result = points_to_ts_indices(points, image_layer3d)
    assert all(isinstance(p[0], slice) for p in result)
    assert np.allclose(target_no_slice, [p[1:] for p in result])


def test_shape_to_ts_indices_invalid_dim(image_layer3d):
    # Test for error raise upon shape with invalid dimensions
    shape = np.array([[1]])
    with pytest.raises(
        ValueError,
        match="Shape must not have more than one dimension less than layer.",
    ):
        shape_to_ts_indices(shape, image_layer3d)


def test_shape_to_ts_indices_multi_plane(image_layer3d, monkeypatch):
    # Test for error raise upon non uniplanar shape
    def mock_func(points, layer):
        return points

    monkeypatch.setattr(
        "napari_time_series_plotter.utils.to_layer_space", mock_func
    )
    shape = np.array([[1, 2, 3], [3, 2, 3]])
    with pytest.raises(
        ValueError,
        match="All vertices of a shape must be in a single y/x plane.",
    ):
        shape_to_ts_indices(shape, image_layer3d)


def test_shape_to_ts_indices(shape, image_layer3d, monkeypatch):
    # Test conversion of shapes to time series indices
    def mock_func(points, layer):
        return points[:, -layer.ndim :]

    monkeypatch.setattr(
        "napari_time_series_plotter.utils.to_layer_space", mock_func
    )
    rshape = np.round(shape).astype(int)

    # Tests with rectangles
    rect_data = image_layer3d.data[
        :, rshape[0, -2] : rshape[2, -2] + 1, rshape[0, -1] : rshape[2, -1] + 1
    ]
    # filled
    rect_target = (
        rect_data.flatten()
    )  # np.reshape(rect_data, (100, rect_data[0].size))
    result = shape_to_ts_indices(
        shape, image_layer3d, ellipsis=False, filled=True
    )
    assert len(result) == image_layer3d.ndim
    assert isinstance(result[0], slice)
    assert np.allclose(
        sorted(image_layer3d.data[result].flatten()), sorted(rect_target)
    )
    # unfilled
    mask = np.ones_like(rect_data, dtype=bool)
    mask[:, 1:-1, 1:-1] = 0
    rect_edge_data = rect_data[mask]
    result = shape_to_ts_indices(
        shape, image_layer3d, ellipsis=False, filled=False
    )
    assert len(result) == image_layer3d.ndim
    assert isinstance(result[0], slice)
    assert np.allclose(
        sorted(image_layer3d.data[result].flatten()), sorted(rect_edge_data)
    )

    # Test with ellipsis
    # filled
    # TODO: Add assertions similar to rect
    result = shape_to_ts_indices(
        shape, image_layer3d, ellipsis=True, filled=True
    )
    assert len(result) == image_layer3d.ndim
    assert isinstance(result[0], slice)
    # unfilled
    # TODO: Add assertions similar to rect
    result = shape_to_ts_indices(
        shape, image_layer3d, ellipsis=True, filled=False
    )
    assert len(result) == image_layer3d.ndim
    assert isinstance(result[0], slice)


def test_align_value_length():
    # Test all value lists are of equal length
    d = {
        "a": [1.0, 2.0],
        "b": [1.0, 2.0, 3.0],
        "c": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
    aligned_d = align_value_length(d)
    assert all(len(val) == 6 for val in aligned_d.values())


def test_viewitemdelegate():
    # TODO: Add tests for ViewItemDelegate
    pass
