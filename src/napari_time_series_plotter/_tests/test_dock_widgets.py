"""
Napari-time_series_plotter dock_widgets tests.
"""
import napari
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel

from ..dock_widgets import (
    TimeSeriesExplorer,
    TimeSeriesTableView,
)


@pytest.fixture
def explorer(make_napari_viewer):
    viewer = make_napari_viewer(show=False)
    yield TimeSeriesExplorer(viewer)


@pytest.fixture
def tableview(make_napari_viewer):
    viewer = make_napari_viewer(show=False)
    yield TimeSeriesTableView(viewer)


# TimeSeriesExplorer tests
def test_TimeSeriesExplorer(explorer):
    # Test TimeSeriesExplorer init
    assert explorer
    assert explorer._napari_viewer == napari.current_viewer()
    assert explorer.layout()


# TimeEsriesTableView tests
def test_TimeSeriesTableView(tableview):
    # Test TimeSeriesTableView init
    assert tableview
    assert isinstance(tableview.source_model, QStandardItemModel)
    assert tableview.layout()


def test_TimeSeriesTableView_connect_callbacks(
    make_napari_viewer, mocker, qtbot
):
    # Test button clicked signal connections
    viewer = make_napari_viewer(show=False)
    mocker_copyToClipboard = mocker.patch.object(
        TimeSeriesTableView, "_copyToClipboard"
    )
    mocker_exportToCSV = mocker.patch.object(
        TimeSeriesTableView, "_exportToCSV"
    )
    tableview = TimeSeriesTableView(viewer)

    # Copy to clipboard btn
    qtbot.mouseClick(tableview.btn_copy, Qt.MouseButton.LeftButton)
    mocker_copyToClipboard.assert_called()

    # Export to csv btn
    qtbot.mouseClick(tableview.btn_export, Qt.MouseButton.LeftButton)
    mocker_exportToCSV.assert_called()
