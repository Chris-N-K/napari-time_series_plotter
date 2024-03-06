"""
This module contains utility functions of napari-time-series-plotter.
"""
from itertools import zip_longest
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Tuple,
)

import napari
import napari.layers.shapes._shapes_utils as shape_utils
import numpy as np
import numpy.typing as npt
from napari.utils.theme import get_theme
from qtpy.QtCore import (
    QModelIndex,
    Qt,
)
from qtpy.QtGui import (
    QColor,
    QPainter,
)
from qtpy.QtWidgets import (
    QApplication,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QWidget,
)
from skimage.draw import line, polygon


def to_world_space(data: npt.NDArray, layer: Any) -> npt.NDArray:
    """Transform layer point coordinates to viewer world space.

    Data array must contain the points in dim 0 and the
    coordinates per dimension in dim 1. The point ccordinates must
    have the same dimensionality as the parent layer.

    Paramaeters
    -----------
    data : np.ndarray
        Point coordinates in an array.
    layer : subclass of napari.layers.Layer
        Parent layer.

    Returns
    -------
    np.ndarray
        Transformed point coordinates in an array.

    Raises
    ------
    ValueError
        If the point coordinates have a different dimensionality
        than the parent layer.
    """
    if data.size != 0:
        if data.shape[1] != layer.ndim:
            raise ValueError(
                f"Point coordinates have different dimensionality: {data.shape[1]} than parent layer: {layer.ndim}."
            )
        tdata = layer._transforms[1:].simplified(data)
        return tdata
    return data


def to_layer_space(data, layer):
    """Transform world space point coordinates to layer space.

    Data array must contain the points in dim 0 and the
    coordinates per dimension in dim 1.

    Paramaeters
    -----------
    data : np.ndarray
        Point coordinates in an array.
    layer
        Target napari layer.

    Returns
    -------
        Transformed point coordinates in an array.
    """
    if data.size != 0:
        if data.shape[1] > layer.ndim:
            tdata = layer._transforms[1:].simplified.inverse(
                data[:, -layer.ndim :]
            )
        elif data.shape[1] < layer.ndim:
            tdata = (
                layer._transforms[1:]
                .simplified.set_slice(
                    range(layer.ndim - data.shape[1], layer.ndim)
                )
                .inverse(data)
            )
        else:
            tdata = layer._transforms[1:].simplified.inverse(data)
        return tdata
    return data


def points_to_ts_indices(points: npt.NDArray, layer) -> List[Tuple[Any, ...]]:
    """Transform point coordinates to time series indices for a given layer.

    Points can be one dimension smaller than the target layer, because
    the indices will always include the complete first dimension (t).
    Points more than one dimension smaller will cause an error.
    If the points are of higher dimensinality than the target layer the
    additional dimensions are dropped.

    Parameters
    ----------
    points: np.ndarray
        Point coordinates to transform into time series indices.
    layer: subclass of napari.layer.Layers
        Layer to generate time series index for.

    Returns
    -------
    indices : list of indice tuples
        Time series indices.

    Raises
    ------
    ValueError
        If point dimensions are more than on d smaller than layer dimensions.
    """
    # Return empty list is points is empty
    if points.size != 0:
        # Ensure correct dimensionality
        ndim = points.shape[1]
        if ndim < layer.ndim - 1:
            raise ValueError(
                f"Point coordinates can have only one dimension less than layer. Ndim points where: {ndim}, ndim layer was: {layer.ndim}"
            )
        elif ndim == layer.ndim - 1:
            spatial_idx = 0
        else:
            spatial_idx = layer.ndim - 1
        tpoints = np.round(to_layer_space(points, layer)).astype(int)
        indices = [
            (slice(None), *p[-spatial_idx:])
            for p in tpoints
            if all(
                0 <= i < d
                for i, d in zip(
                    p[-spatial_idx:], layer.data.shape[-spatial_idx:]
                )
            )
        ]
    else:
        indices = []
    return indices


def shape_to_ts_indices(
    data: npt.NDArray,
    layer: napari.layers.Image,
    ellipsis: bool = False,
    filled: bool = True,
) -> Tuple[Any, ...]:
    """Transform a shapes face or edges to time series indices for a given layer.

    Shape data must be of same or bigger dimensionality as layer or maximal one smaller,
    as the returned indices include the full first dimension (t).
    Shape data with less dimensions or non-uniplanar shapes will raise an error.

    Parameters
    ----------
    data: npndarray
        Shape object data to transform into time series indices.
    layer: napari.layers.Image
        Image layer to generate time series index for.
    ellipsis: bool
        If true triangulated ellipsis vertices and generate indices from them instead.
    filled: bool
        If true return indices for the filled shape, else only for the edges.

    Returns
    ------
    ts_indices : tuple of tuples
        Indices in form of a nested tuple with layer.ndim elements.
        The first element is a slice over the full first dimension.
        The following tuples have the same number of elements encoding
        voxel positions.

    Raises
    ------
    ValueError
        If shape hase more than one dimension less than the target layer.
    ValueError
        If the shape is not y/x planar.
    ValueError
        If data is collinear.
    """
    # ensure correct dimensionality
    ndim = data.shape[1]
    if ndim < layer.ndim - 1:
        raise ValueError(
            f"Shape must not have more than one dimension less than layer. Ndim shape was: {ndim}, ndim layer was: {layer.ndim} "
        )

    tdata = np.round(to_layer_space(data, layer)).astype(int)

    # ensure shape is y/x planar
    if len(np.unique(tdata[:, :-2], axis=0)) == 1:
        val = np.expand_dims(np.round(tdata[0, 1:-2]).astype(int), axis=0)
    else:
        raise ValueError(
            "All vertices of a shape must be in a single y/x plane."
        )

    # determine vertices
    if ellipsis:
        if shape_utils.is_collinear(tdata[:, -2:]):
            raise ValueError("Shape data must not be collinear.")
        else:
            vertices, _ = shape_utils.triangulate_ellipse(tdata[:, -2:])
    else:
        vertices = tdata[:, -2:]

    # determine y/x indices from vertices
    if filled:
        raw_idx = polygon(
            vertices[:, 0], vertices[:, 1], layer.data.shape[-2:]
        )
    else:
        vertices = np.round(
            np.clip(vertices, 0, np.asarray(layer.data.shape[-2:]) - 1)
        ).astype(int)
        raw_idx = [[], []]
        for v1, v2 in zip_longest(
            vertices, vertices[1:], fillvalue=vertices[0]
        ):
            y, x = line(*v1, *v2)
            raw_idx[0].extend(y[:-1])
            raw_idx[1].extend(x[:-1])

    # drop duplicate indices
    cleaned_idx = tuple(map(tuple, np.unique(raw_idx, axis=0)))

    # expand indices to full dimensions
    exp = tuple(np.repeat(val, len(cleaned_idx[0]), axis=0).T)
    ts_indices = (slice(None),) + exp + cleaned_idx

    return ts_indices


def align_value_length(
    dictionary: Dict[Any, Collection]
) -> Dict[Any, npt.NDArray]:
    """Align the lengths of dictionary values by appending NaNs.

    Parameters
    ----------
    dictionary : Dict[Any, Collection]
        Dictionary to process.

    Returns
    -------
    matched : Dict[Any, Collection]
        Dictionary with aligned value lengths.
    """
    max_len = np.max([len(val) for val in dictionary.values()])
    matched = {}
    for key, val in dictionary.items():
        if len(val) < max_len:
            pval = np.full((max_len), np.nan)
            pval[: len(val)] = val
        else:
            pval = val
        matched[key] = pval
    return matched


class ViewItemDelegate(QStyledItemDelegate):
    """ItemDelegate to style LayerSelector items.

    Only if a napari.Viewer instance is present customization is executed.
    The delegate customizes the checkbox to match the viewer text color.

    Parameters
    ----------
    parent : qtpy.QtWidgets.QWidget
        Parent widget.

    Methods
    -------
    paint(painter=QPainter, option=QStyleOptionViewItem, index=QModelIndex)
        Renders the delegate using the given painter and style option for the item specified by index.
    """

    def __init__(self, parent: QWidget = None) -> None:
        super().__init__(parent=parent)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> None:
        """Renders the delegate using the given painter and style option for the item specified by index.

        Parameters
        ----------
        painter: QPainter
            Painter to draw the item.
        option: QStyleOptionViewItem
            Options defining the item style.
        index: QModelIndex
            Item index.
        """
        super().paint(painter, option, index)
        viewer = napari.current_viewer()
        if viewer:
            self.initStyleOption(option, index)
            painter.setPen(
                QColor(get_theme(viewer.theme, as_dict=False).text.as_hex())
            )

            widget = option.widget
            style = widget.style() if widget else QApplication.style()
            check_rect = style.subElementRect(
                QStyle.SubElement.SE_ItemViewItemCheckIndicator,
                option,
                widget,
            )
            if index.data(Qt.ItemDataRole.CheckStateRole) == Qt.Checked:
                painter.drawRect(check_rect)
                style.drawPrimitive(
                    QStyle.PrimitiveElement.PE_IndicatorCheckBox,
                    option,
                    painter,
                    widget,
                )
            else:
                painter.drawRect(check_rect)
