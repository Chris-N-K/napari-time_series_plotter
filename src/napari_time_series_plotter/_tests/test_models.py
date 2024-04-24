import napari
import numpy as np
import pytest
from qtpy.QtCore import Qt

from ..models import (
    LayerItem,
    LayerSelectionModel,
    LivePlotItem,
    SelectionLayerItem,
    SourceLayerItem,
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
def napari_viewer(make_napari_viewer):
    viewer = make_napari_viewer(show=False)
    viewer.theme = "dark"

    img = np.random.random((10, 50, 50, 50))
    viewer.add_image(img[0, 0, ...], name="img2D")
    viewer.add_image(img[0, ...], name="img3D")
    viewer.add_image(img, name="img4D")

    points = np.random.uniform(0, 50, (10, 4))
    viewer.add_points(points[:, 2:], name="points2D")
    viewer.add_points(points[:, 1:], name="points3D")

    shapes = np.array(
        [
            [
                [5.0, 32.22085985, 32.82546271],
                [5.0, 8.73997325, 46.08423441],
                [5.0, 33.37084906, 39.67381897],
                [5.0, 40.93833238, 45.90068874],
            ],
            [
                [5.0, 7.22979874, 28.11437405],
                [5.0, 13.41083848, 23.97879352],
                [5.0, 5.1118159, 11.49239297],
                [5.0, 3.08541644, 45.40826843],
            ],
            [
                [5.0, 4.01948763, 5.96978842],
                [5.0, 16.5139378, 26.88169288],
                [5.0, 26.61494904, 39.66879031],
                [5.0, 31.11766735, 27.74837614],
            ],
            [
                [5.0, 38.52456861, 2.63444864],
                [5.0, 33.63103407, 17.87269485],
                [5.0, 17.0881018, 18.03530914],
                [5.0, 7.00764648, 8.17156158],
            ],
            [
                [5.0, 15.41283804, 16.77159509],
                [5.0, 9.78001702, 9.30676253],
                [5.0, 9.80606786, 38.36315954],
                [5.0, 35.69650342, 18.78553517],
            ],
            [
                [5.0, 25.02243595, 30.98579199],
                [5.0, 13.16157468, 27.74830907],
                [5.0, 40.61275848, 19.52976674],
                [5.0, 24.62203594, 48.00278828],
            ],
            [
                [5.0, 19.56785736, 36.45052502],
                [5.0, 1.09854107, 44.89968895],
                [5.0, 10.60803909, 29.98479221],
                [5.0, 10.73179309, 21.01183818],
            ],
            [
                [5.0, 1.69914104, 18.87369463],
                [5.0, 20.64974713, 20.8164179],
                [5.0, 10.34259419, 24.59212866],
                [5.0, 8.12208014, 29.02975358],
            ],
            [
                [5.0, 45.32032118, 18.97454619],
                [5.0, 27.02443551, 31.58639186],
                [5.0, 12.91434733, 43.43764464],
                [5.0, 12.30973863, 48.00278828],
            ],
            [
                [5.0, 43.17424602, 32.66731867],
                [5.0, 39.03643289, 24.47929341],
                [5.0, 39.65105704, 4.44448297],
                [5.0, 19.46532318, 31.2729772],
            ],
        ]
    )
    viewer.add_shapes(shapes[:, :, 1:], name="shapes2D")
    viewer.add_shapes(shapes, name="shapes3D")
    yield viewer


@pytest.fixture
def napari_event():
    class MockEvent:
        def __init__(self):
            self.value = None
            self.modifiers = None

    yield MockEvent()


@pytest.mark.skip(reason="")
def test_LayerItem(napari_viewer):
    # Test LayerItem initialization
    items = [LayerItem(layer) for layer in napari_viewer.layers]

    for item, layer in zip(items, napari_viewer.layers):
        assert item._layer == layer
        assert (
            item.foreground().color().name()
            == napari.utils.theme.get_theme(
                napari_viewer.theme, as_dict=False
            ).text.as_hex()
        )

    # Test theme change callback connection
    napari_viewer.theme = "light"
    for item in items:
        assert (
            item.foreground().color().name()
            == napari.utils.theme.get_theme(
                "light", as_dict=False
            ).text.as_hex()
        )


@pytest.mark.skip(reason="")
def test_LayerItem_data():
    # Test data method returns
    layer = napari.layers.Image(np.random.random((10, 100, 100)), name="Image")
    item = LayerItem(layer)

    assert item.data(Qt.DisplayRole) == layer.name
    assert item.data(Qt.UserRole + 1) == layer._type_string
    assert item.data(Qt.UserRole + 2) == layer


@pytest.mark.skip(reason="")
def test_LayerItem_isChecked():
    # Test for checkstate return
    layer = napari.layers.Image(np.random.random((10, 100, 100)), name="Image")
    item = LayerItem(layer)
    assert not item.isChecked()

    item.setCheckState(Qt.Checked)
    assert item.isChecked()


@pytest.mark.skip(reason="")
def test_SourceLayerItem_children(napari_viewer):
    # Test return of child items
    item = SourceLayerItem(napari_viewer.layers["img3D"])
    children = [
        SourceLayerItem(napari_viewer.layers["img2D"]),
        SourceLayerItem(napari_viewer.layers["img4D"]),
    ]
    assert not item.children()

    item.appendRows(children)
    assert item.children() == children


@pytest.mark.skip(reason="")
def test_SourceLayerItem_findChildren(napari_viewer):
    # Test child search by value
    item = SourceLayerItem(napari_viewer.layers["img3D"])
    children = [
        SourceLayerItem(napari_viewer.layers["img2D"]),
        SourceLayerItem(napari_viewer.layers["img4D"]),
    ]
    assert not item.findChildren("img4D", Qt.DisplayRole)

    item.appendRows(children)
    assert item.findChildren("img2D", Qt.DisplayRole) == children[:1]
    assert item.findChildren("img4D", Qt.DisplayRole) == children[1:]


@pytest.mark.skip(reason="")
def test_SelectionLayerItem_init(napari_viewer, mocker):
    layers = napari_viewer.layers

    mocker.patch.object(SelectionLayerItem, "_extract_indices")
    mocker.patch.object(SelectionLayerItem, "_extract_ts_data")
    mocker.patch.object(SelectionLayerItem, "_connect_callbacks")

    # Test initialization with valid layer types
    p2d = SourceLayerItem(layers["img2D"])
    p3d = SourceLayerItem(layers["img3D"])
    p4d = SourceLayerItem(layers["img4D"])
    items = [
        SelectionLayerItem(layers["points3D"], p2d),
        SelectionLayerItem(layers["points3D"], p3d),
        SelectionLayerItem(layers["points3D"], p4d),
        SelectionLayerItem(layers["shapes3D"], p2d),
        SelectionLayerItem(layers["shapes3D"], p3d),
        SelectionLayerItem(layers["shapes3D"], p4d),
    ]
    for item in items:
        assert isinstance(
            item._layer, (napari.layers.Points, napari.layers.Shapes)
        )
        assert isinstance(item._parent, SourceLayerItem)

    # Test initialization with invalid layer type
    with pytest.raises(ValueError, match="must be of type Points or Shapes"):
        SelectionLayerItem(layers["img3D"], layers["img4D"])

    # Test with invalid layer size
    with pytest.raises(
        ValueError, match="must not be more than one dimension smaller"
    ):
        SelectionLayerItem(layers["points2D"], p4d)
    with pytest.raises(
        ValueError, match="must not be more than one dimension smaller"
    ):
        SelectionLayerItem(layers["shapes2D"], p4d)


@pytest.mark.skip(reason="")
def test_SelectionLayerItem_connect_callbacks(napari_viewer, mocker):
    layer = napari_viewer.layers["points3D"]
    parent_layer = napari_viewer.layers["img3D"]
    new_data = np.array([[1, 2, 3], [4, 5, 6]])

    mocker.patch.object(SelectionLayerItem, "_extract_indices")
    mocker.patch.object(SelectionLayerItem, "_extract_ts_data")
    m1 = mocker.patch.object(SelectionLayerItem, "updateTSIndices")
    m2 = mocker.patch.object(SelectionLayerItem, "updateTSData")

    SelectionLayerItem(layer, SourceLayerItem(parent_layer))

    # Test layer event callback
    layer.data = new_data
    m1.assert_called()

    # Test parent layer event callback
    parent_layer.data = new_data
    m2.assert_called()


@pytest.mark.skip(reason="")
def test_SelectionLayerItem_extract_indices_points(napari_viewer, mocker):
    # Test indice extraction from points layers
    mocker.patch.object(SelectionLayerItem, "_extract_ts_data")

    # 2D points layer
    points2d = napari_viewer.layers["points2D"]
    points_selection2d = SelectionLayerItem(
        points2d, SourceLayerItem(napari_viewer.layers["img3D"])
    )
    assert np.array_equal(
        np.asarray(points_selection2d._indices)[:, 1:], np.round(points2d.data)
    )

    # 3D points layer
    points3d = napari_viewer.layers["points3D"]
    points_selection3d = SelectionLayerItem(
        points3d, SourceLayerItem(napari_viewer.layers["img3D"])
    )
    assert np.array_equal(
        np.asarray(points_selection3d._indices)[:, 1:],
        np.round(points3d.data)[:, 1:],
    )

    # Extraction from empty layer
    empty = napari.layers.Points()
    empty_selection = SelectionLayerItem(
        empty, SourceLayerItem(napari_viewer.layers["img3D"])
    )
    assert not empty_selection._indices


@pytest.mark.skip(reason="")
def test_SelectionLayerItem_extract_indices_shapes(napari_viewer, mocker):
    # Test indice extraction from shapes layers
    mocker.patch.object(SelectionLayerItem, "_extract_ts_data")

    # 2D shapes layer
    shapes2d = napari_viewer.layers["shapes2D"]
    shapes_selection2d = SelectionLayerItem(
        shapes2d, SourceLayerItem(napari_viewer.layers["img3D"])
    )
    assert len(shapes_selection2d._indices) == 10
    # TODO: more sophisticated comparisons would be good

    # 3D shapes layer
    shapes3d = napari_viewer.layers["shapes3D"]
    shapes_selection3d = SelectionLayerItem(
        shapes3d, SourceLayerItem(napari_viewer.layers["img3D"])
    )
    assert len(shapes_selection3d._indices) == 10

    # Extraction from empty layer
    empty = napari.layers.Shapes()
    empty_selection = SelectionLayerItem(
        empty, SourceLayerItem(napari_viewer.layers["img3D"])
    )
    assert not empty_selection._indices


@pytest.mark.skip(reason="")
def test_SelectionLayerItem_extract_ts_data(napari_viewer):
    # Test data extraction with points selection layer from source layer
    layer = napari_viewer.layers["img3D"]
    indices = [
        (
            slice(None, None, None),
            np.random.randint(0, 25, (10)),
            np.random.randint(0, 25, (10)),
        ),
        (
            slice(None, None, None),
            np.random.randint(0, 50, (10)),
            np.random.randint(0, 50, (10)),
        ),
    ]

    # With no selection
    item = SelectionLayerItem(napari.layers.Points(), SourceLayerItem(layer))
    assert not item._ts_data

    # With selection
    item._indices = indices
    assert np.array_equal(
        item._extract_ts_data()[0], layer.data[indices[0]]
    ) & np.array_equal(item._extract_ts_data()[1], layer.data[indices[1]])


@pytest.mark.skip(reason="")
def test_SelectionLayerItem_data():
    # Test data() return values
    item = SelectionLayerItem(
        napari.layers.Points(),
        SourceLayerItem(napari.layers.Image(np.array([[1]]))),
    )

    # ID
    assert "Points" in item.data(Qt.UserRole + 5) and "Image" in item.data(
        Qt.UserRole + 5
    )


@pytest.mark.skip(reason="")
def test_SelectionLayerItem_updateTSIndices(napari_viewer, mocker):
    # Test if TS data is updated if indices changed
    mocker.patch.object(SelectionLayerItem, "_connect_callbacks")
    m1 = mocker.patch.object(SelectionLayerItem, "updateTSData")
    source_item = SourceLayerItem(napari_viewer.layers["img3D"])
    points_layer = napari_viewer.layers["points3D"]
    shapes_layer = napari_viewer.layers["shapes3D"]
    points_selection_item = SelectionLayerItem(points_layer, source_item)
    shapes_selection_item = SelectionLayerItem(shapes_layer, source_item)

    # Check for skip if no indice changes
    points_selection_item.updateTSIndices()
    shapes_selection_item.updateTSIndices()
    m1.assert_not_called()

    # Update if indices changed
    new_points3d = np.random.uniform(0, 25, (10, 3))
    points_layer.data = new_points3d
    points_selection_item.updateTSIndices()
    m1.assert_called()
    assert np.array_equal(
        np.asarray(points_selection_item._indices)[:, 1:],
        np.round(new_points3d.data)[:, 1:],
    )

    new_shapes3d = np.array(
        [
            [
                [1, 5, 5],
                [1, 5, 10],
                [1, 10, 10],
                [1, 10, 5],
            ],
        ]
    )
    shapes_layer.data = new_shapes3d
    shapes_selection_item.updateTSIndices()
    assert m1.call_count == 2
    new_indices = np.asarray(shapes_selection_item._indices[0][1:])
    assert (
        np.all(new_indices >= 5) & np.all(new_indices <= 10)
        and new_indices.shape[1] == 6**2
    )


@pytest.mark.skip(reason="")
def test_LivePlotItem_extract_ts_data(napari_viewer, mocker):
    # Test TS data extraction from all valid items
    class MockModel:
        def __init__(self):
            self.items = [
                SourceLayerItem(
                    napari.layers.Image(np.array([[1]]), name="LivePlot")
                ),
                SourceLayerItem(
                    napari.layers.Image(np.array([[1]]), name="empty")
                ),
                SourceLayerItem(napari_viewer.layers["img3D"]),
                SourceLayerItem(napari_viewer.layers["img3D"]),
                SourceLayerItem(napari_viewer.layers["img4D"]),
            ]
            self.items[2].setCheckState(Qt.Unchecked)

        def item(self, row, col):
            return self.items[row]

        def rowCount(self):
            return len(self.items)

    def model(self):
        return MockModel()

    mocker.patch.object(LivePlotItem, "model", new=model)
    live_plot_item = LivePlotItem()
    live_plot_item._cpos = (0, 0, 1, 1)
    res = live_plot_item._extract_ts_data()
    assert len(res) == 2
    assert all([len(res[0]) == 50, len(res[1] == 10)])


@pytest.mark.skip(reason="")
def test_LayerSelectionModel(napari_viewer, mocker, qtbot):
    # Test LayerSelectionModel initialization.
    mocker.patch.object(LayerSelectionModel, "_init_data")
    m1 = mocker.patch.object(LayerSelectionModel, "_layer_inserted_callback")
    m2 = mocker.patch.object(LayerSelectionModel, "_layer_removed_callback")
    m3 = mocker.patch.object(LayerSelectionModel, "_mouse_move_callback")

    # Test with agg_func argument.
    model = LayerSelectionModel(napari_viewer, agg_func=np.sum)
    assert model._agg_func == np.sum

    # Test ithout agg_func argument.
    model = LayerSelectionModel(napari_viewer)
    assert model._agg_func == np.mean

    # Test for name change event detection.
    with qtbot.waitSignal(model.dataChanged):
        napari_viewer.layers["img3D"].name = "test"

    # Test for layer inserted event detection.
    napari_viewer.add_image(np.array([[1]]))
    m1.assert_called()

    # Test for layer removed event detetction.
    napari_viewer.layers.pop()
    m2.assert_called()

    # Test for mouse move callback registration.
    assert m3 in napari_viewer.mouse_move_callbacks


@pytest.mark.skip(reason="")
def test_LayerSelectionModel_init_data(napari_viewer, mocker, qtbot):
    # Test model data initialization.
    mocker.patch.object(LivePlotItem, "updateTSData")
    napari_viewer.add_image(np.array([[[1, 2, 3]]]), rgb=True)

    model = LayerSelectionModel(napari_viewer)

    assert model.rowCount() == 3
    assert isinstance(model.item(0), LivePlotItem)
    assert len(model.item(1).children()) == 2
    assert len(model.item(2).children()) == 4

    # Test if signal is emited after initialization
    with qtbot.waitSignal(model.dataChanged):
        model._init_data(napari_viewer.layers)


@pytest.mark.skip(reason="")
def test_LayerSelectionModel_layer_inserted_callback(
    napari_viewer, napari_event, mocker
):
    # Test callback for napari LayerList layer inserted events.
    m1 = mocker.patch.object(LivePlotItem, "updateTSData")
    model = LayerSelectionModel(napari_viewer)
    assert model.rowCount() == 3

    # Test for invalid layer type.
    napari_event.value = napari.layers.Labels(np.array([[0]]))
    model._layer_inserted_callback(napari_event)
    assert model.rowCount() == 3

    # Test for Image layer insertion.
    napari_event.value = napari.layers.Image(np.array([[0]]), name="newImg2D")
    model._layer_inserted_callback(napari_event)
    assert not model.findItems("newImg2D")
    napari_event.value = napari.layers.Image(
        np.array([[[1, 2, 3]]]), name="newImgRGB", rgb=True
    )
    model._layer_inserted_callback(napari_event)
    assert not model.findItems("newImgRGB")
    napari_event.value = napari.layers.Image(
        np.array([[[0]]]), name="newImgValid"
    )
    model._layer_inserted_callback(napari_event)
    assert len(model.findItems("newImgValid")) == 1
    assert model.item(1).data() == "newImgValid"
    assert len(model.item(1).children()) == 4
    assert isinstance(model.item(0), LivePlotItem)

    # Test for Points layer insertion.
    napari_event.value = napari.layers.Points(
        np.array([[0, 0]]), name="newPoints"
    )
    model._layer_inserted_callback(napari_event)
    assert (
        (len(model.item(1).findChildren("newPoints")) == 1)
        & (len(model.item(2).findChildren("newPoints")) == 0)
        & (len(model.item(3).findChildren("newPoints")) == 1)
    )

    # Test for Shapes layer insertion.
    napari_event.value = napari.layers.Shapes(
        np.array([[[0, 0], [0, 1], [1, 1], [1, 0]]]), name="newShapes"
    )
    model._layer_inserted_callback(napari_event)
    assert (
        (len(model.item(1).findChildren("newShapes")) == 1)
        & (len(model.item(2).findChildren("newShapes")) == 0)
        & (len(model.item(3).findChildren("newShapes")) == 1)
    )

    # Check if all valid event values triggered ts data update.
    assert m1.call_count == 3


@pytest.mark.skip(reason="")
def test_LayerSelectionModel_layer_removed_callback(
    napari_viewer, napari_event, mocker
):
    # Test callback for napari LayerList layer removed events.
    m1 = mocker.patch.object(LivePlotItem, "updateTSData")
    model = LayerSelectionModel(napari_viewer)
    assert model.rowCount() == 3

    # Test with layer not in model.
    napari_event.value = napari_viewer.layers["img2D"]
    model._layer_removed_callback(napari_event)
    assert model.rowCount() == 3

    # Test removal of SelectionLayerItem.
    napari_event.value = napari_viewer.layers["points3D"]
    model._layer_removed_callback(napari_event)
    assert (len(model.item(1).children()) == 1) & (
        len(model.item(2).children()) == 3
    )

    # Test removal of SourceLayerItem.
    napari_event.value = napari_viewer.layers["img3D"]
    model._layer_removed_callback(napari_event)
    assert model.rowCount() == 2

    # Check if all valid event values triggered ts data update.
    assert m1.call_count == 2


@pytest.mark.skip(reason="")
def test_LayerSelectionModel_mouse_move_callback(
    napari_viewer, napari_event, mocker
):
    # Test calback for napari mouse move events.
    m1 = mocker.patch.object(LivePlotItem, "setCpos")
    napari_viewer.cursor.position = (0, 0, 0, 0)
    model = LayerSelectionModel(napari_viewer)

    # Event without 'Shift' modifier
    napari_event.modifiers = []
    model._mouse_move_callback(napari_viewer, napari_event)
    m1.assert_not_called()

    # Event with 'Shift' modifier
    napari_event.modifiers = ["Shift"]
    cpos = tuple(np.random.randint(0, 10, 4))
    napari_viewer.cursor.position = cpos
    model._mouse_move_callback(napari_viewer, napari_event)
    m1.assert_called_once_with(cpos)


def test_LayerSelectionModel_tsData(napari_viewer, mocker):
    def mock_return1(self, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return "LivePlot"
        elif Qt.UserRole + 4:
            return [4]
        elif role == Qt.UserRole + 5:
            return "LivePlot"

    mocker.patch.object(LivePlotItem, "data", new=mock_return1)

    model = LayerSelectionModel(napari_viewer)
    retval = model.tsData
    assert retval
    raise ValueError(retval)
