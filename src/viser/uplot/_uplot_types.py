"""Clean uPlot TypedDict definitions."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Literal
from typing_extensions import Never, Required, TypedDict

# Semantic type aliases for unsupported/complex TypeScript patterns
JSCallback = Never  # JavaScript function signatures
DOMElement = Never  # DOM elements (HTMLElement, etc.)
CSSValue = str  # CSS property values
UnknownType = Any  # Unknown interface references


# Enum definitions
class JoinNullMode(IntEnum):
    REMOVE = 0
    RETAIN = 1
    EXPAND = 2


class Orientation(IntEnum):
    HORIZONTAL = 0
    VERTICAL = 1


class Mode(IntEnum):
    ALIGNED = 1
    FACETED = 2


class FocusBias(IntEnum):
    NONE = 0
    AWAY_FROM_ZERO = 1
    TOWARDS_ZERO = -1


class HoverBias(IntEnum):
    NONE = 0
    FORWARD = 1
    BACKWARD = -1


class Distr(IntEnum):
    LINEAR = 1
    ORDINAL = 2
    LOGARITHMIC = 3
    ARC_SINH = 4
    CUSTOM = 100


class BarsPathBuilderFacetUnit(IntEnum):
    SCALE_VALUE = 1
    PIXEL_PERCENT = 2
    COLOR = 3


class BarsPathBuilderFacetKind(IntEnum):
    UNARY = 1
    DISCRETE = 2
    CONTINUOUS = 3


class Sorted(IntEnum):
    UNSORTED = 0
    ASCENDING = 1
    DESCENDING = -1


class Side(IntEnum):
    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3


class Align(IntEnum):
    LEFT = 1
    RIGHT = 2


class AlignTo(IntEnum):
    INSIDE = 1
    OUTSIDE = 2


# Type aliases for index signatures
# Index signature: key -> Scale.
Scales = Dict[str, "Scale"]

Legend_Markers = TypedDict(
    "Legend_Markers",
    {
        "show": bool,
        # series indicator line width.
        "width": float | JSCallback,
        # series indicator stroke (CSS borderColor).
        "stroke": CSSValue,
        # series indicator fill.
        "fill": CSSValue,
        # series indicator stroke style (CSS borderStyle).
        "dash": CSSValue,
    },
    total=False,
)

Focus = TypedDict(
    "Focus",
    {
        # alpha-transparancy of de-focused series.
        "alpha": float
    },
)

BBox = TypedDict(
    "BBox",
    {
        "left": Required[float],
        "top": Required[float],
        "width": Required[float],
        "height": Required[float],
        "show": bool,
    },
    total=False,
)

Select = TypedDict(
    "Select",
    {
        # div into which .u-select will be placed: .u-over or .u-under.
        "over": bool
    },
    total=False,
)

Cursor_Bind = TypedDict(
    "Cursor_Bind",
    {
        "mousedown": UnknownType,
        "mouseup": UnknownType,
        "click": UnknownType,
        "dblclick": UnknownType,
        "mousemove": UnknownType,
        "mouseleave": UnknownType,
        "mouseenter": UnknownType,
    },
    total=False,
)

Cursor_Drag = TypedDict(
    "Cursor_Drag",
    {
        "setScale": bool,
        # toggles dragging along x.
        "x": bool,
        # toggles dragging along y.
        "y": bool,
        # min drag distance threshold.
        "dist": float,
        # when x & y are true, sets an upper drag limit in CSS px for adaptive/unidirectional behavior.
        "uni": float,
        # post-drag "click" event proxy, default is to prevent these click events.
        "click": UnknownType,
    },
    total=False,
)

Cursor_Hover = TypedDict(
    "Cursor_Hover",
    {
        # minimum cursor proximity to datapoint in CSS pixels for point hover.
        "prox": float | None | UnknownType,
        # when non-zero, will only proximity-test indices forward or backward.
        "bias": HoverBias,
        # what values to treat as non-hoverable and trigger scanning to another index.
        "skip": list[Any],
    },
    total=False,
)

Cursor_Focus = TypedDict(
    "Cursor_Focus",
    {
        # minimum cursor proximity to datapoint in CSS pixels for focus activation, disabled: < 0, enabled: <= 1e6.
        "prox": Required[float],
        # when non-zero, will only focus next series towards or away from zero.
        "bias": FocusBias,
        # measures cursor y distance to a series in CSS pixels (for triggering setSeries hook with closest).
        "dist": UnknownType,
    },
    total=False,
)

Scale = TypedDict(
    "Scale",
    {
        # is this scale temporal, with series' data in UNIX timestamps?
        "time": bool,
        # determines whether all series' data on this scale will be scanned to find the full min/max range.
        "auto": bool | JSCallback,
        # can define a static scale range or re-range an initially-determined range from series data.
        "range": tuple[float | None, float | None] | JSCallback | UnknownType,
        # scale key from which this scale is derived.
        "from": str,
        # scale distribution. 1: linear, 2: ordinal, 3: logarithmic, 4: arcsinh.
        "distr": UnknownType,
        # logarithmic base.
        "log": Literal[10] | Literal[2],
        # clamps log scale values <= 0 (default = scaleMin / 10).
        "clamp": float | JSCallback,
        # arcsinh linear threshold.
        "asinh": float,
        # forward transform fn, with custom distr: 100.
        "fwd": JSCallback,
        # backward transform fn, with custom distr: 100.
        "bwd": JSCallback,
        # current min scale value.
        "min": float,
        # current max scale value.
        "max": float,
        # scale direction.
        "dir": Literal[1] | Literal[-1],
        # scale orientation - 0: hz, 1: vt.
        "ori": Literal[0] | Literal[1],
        # own key (for read-back).
        "key": str,
    },
    total=False,
)

Series_Points = TypedDict(
    "Series_Points",
    {
        # if boolean or returns boolean, round points are drawn with defined options, else fn should draw own custom points via self.ctx.
        "show": UnknownType,
        "paths": UnknownType,
        # may return an array of points indices to draw.
        "filter": UnknownType,
        # diameter of point in CSS pixels.
        "size": float,
        # minimum avg space between point centers before they're shown (default: size * 2).
        "space": float,
        # line width of circle outline in CSS pixels.
        "width": float,
        # line color of circle outline (defaults to series.stroke).
        "stroke": UnknownType,
        # line dash segment array.
        "dash": list[float],
        # line cap.
        "cap": CSSValue,
        # fill color of circle (defaults to #fff).
        "fill": UnknownType,
    },
    total=False,
)

Series_Facet = TypedDict(
    "Series_Facet",
    {"scale": Required[str], "auto": bool, "sorted": Sorted},
    total=False,
)

Band = TypedDict(
    "Band",
    {
        # series indices of upper and lower band edges.
        "series": Required[tuple[float, float]],
        # area fill style.
        "fill": CSSValue,
        # whether to fill towards yMin (-1) or yMax (+1) between "from" & "to" series.
        "dir": Literal[1] | Literal[-1],
    },
    total=False,
)

Axis_Border = TypedDict("Axis_Border", {}, total=False)

Axis_Grid = TypedDict("Axis_Grid", {}, total=False)

Axis_Ticks = TypedDict(
    "Axis_Ticks",
    {
        # length of tick in CSS pixels.
        "size": float
    },
    total=False,
)

Plugin = TypedDict(
    "Plugin",
    {
        "hooks": Required[UnknownType],
        # can mutate provided opts as necessary.
        "opts": UnknownType,
    },
    total=False,
)

Cursor_Sync = TypedDict(
    "Cursor_Sync",
    {
        # sync key must match between all charts in a synced group.
        "key": Required[str],
        # determines if series toggling and focus via cursor is synced across charts.
        "setSeries": bool,
        # sets the x and y scales to sync by values. null will sync by relative (%) position.
        "scales": UnknownType,
        # fns that match x and y scale keys and seriesIdxs between publisher and subscriber.
        "match": UnknownType,
        # event filters.
        "filters": UnknownType,
        # sync scales' values at the cursor position (exposed for read-back by subscribers).
        "values": UnknownType,
    },
    total=False,
)

Legend = TypedDict(
    "Legend",
    {
        "show": bool,
        # show series values at current cursor.idx.
        "live": bool,
        # switches primary interaction mode to toggle-one/toggle-all.
        "isolate": bool,
        # series indicators.
        "markers": Legend_Markers,
        # callback for moving the legend elsewhere. e.g. external DOM container.
        "mount": UnknownType,
        # current index (readback-only, not for init).
        "idx": float | None,
        # current indices (readback-only, not for init).
        "idxs": list[float | None],
        # current values (readback-only, not for init).
        "values": list[UnknownType],
    },
    total=False,
)

Cursor_Points = TypedDict(
    "Cursor_Points",
    {
        "show": UnknownType,
        # only show single y-closest point on hover (only works when cursor.focus.prox >= 0).
        "one": bool,
        # hover point diameter in CSS pixels.
        "size": UnknownType,
        # hover point bbox in CSS pixels (will be used instead of size).
        "bbox": UnknownType,
        # hover point outline width in CSS pixels.
        "width": UnknownType,
        # hover point outline color, pattern or gradient.
        "stroke": UnknownType,
        # hover point fill color, pattern or gradient.
        "fill": UnknownType,
    },
    total=False,
)

Series = TypedDict(
    "Series",
    {
        # series on/off. when off, it will not affect its scale.
        "show": bool,
        # className to add to legend parts and cursor hover points.
        "class": str,
        # scale key.
        "scale": str,
        # whether this series' data is scanned during auto-ranging of its scale.
        "auto": bool,
        # if & how the data is pre-sorted (scale.auto optimization).
        "sorted": UnknownType,
        # when true, null data values will not cause line breaks.
        "spanGaps": bool,
        # may mutate and/or augment gaps array found from null values.
        "gaps": UnknownType | JSCallback,
        # whether path and point drawing should offset canvas to try drawing crisp lines.
        "pxAlign": float | bool,
        # legend label.
        "label": str | DOMElement,
        # inline-legend value formatter. can be an fmtDate formatting string when scale.time: true.
        "value": str | JSCallback,
        # table-legend multi-values formatter.
        "values": JSCallback,
        "paths": JSCallback,
        # rendered datapoints.
        "points": Series_Points,
        # facets.
        "facets": list[Series_Facet],
        # line width in CSS pixels.
        "width": float,
        # line & legend color.
        "stroke": CSSValue,
        # area fill & legend color.
        "fill": CSSValue,
        # area fill baseline (default: 0).
        "fillTo": float | JSCallback,
        # line dash segment array.
        "dash": list[float],
        # line cap.
        "cap": CSSValue,
        # alpha-transparancy.
        "alpha": float,
        # current min and max data indices rendered.
        "idxs": tuple[float, float],
        # current min rendered value.
        "min": float,
        # current max rendered value.
        "max": float,
    },
    total=False,
)

Axis = TypedDict(
    "Axis",
    {
        # axis on/off.
        "show": bool,
        # scale key.
        "scale": str,
        # side of chart - 0: top, 1: rgt, 2: btm, 3: lft.
        "side": UnknownType,
        # height of x axis or width of y axis in CSS pixels alloted for values, gap & ticks, but excluding axis label.
        "size": float | JSCallback,
        # gap between axis values and axis baseline (or ticks, if enabled) in CSS pixels.
        "gap": float,
        # font used for axis values.
        "font": CSSValue,
        # font-size multiplier for multi-line axis values (similar to CSS line-height: 1.5em).
        "lineGap": float,
        # color of axis label & values.
        "stroke": CSSValue,
        # axis label text.
        "label": str | JSCallback,
        # height of x axis label or width of y axis label in CSS pixels alloted for label text + labelGap.
        "labelSize": float,
        # gap between label baseline and tick values in CSS pixels.
        "labelGap": float,
        # font used for axis label.
        "labelFont": CSSValue,
        # minimum grid & tick spacing in CSS pixels.
        "space": float | JSCallback,
        # available divisors for axis ticks, values, grid.
        "incrs": list[float] | JSCallback,
        # determines how and where the axis must be split for placing ticks, values, grid.
        "splits": list[float] | JSCallback,
        # can filter which splits are passed to axis.values() for rendering. e.g splits.map(v => v % 2 == 0 ? v : null).
        "filter": JSCallback,
        # formats values for rendering.
        "values": UnknownType,
        # values rotation in degrees off horizontal (only bottom axes w/ side: 2).
        "rotate": float | JSCallback,
        # text alignment of axis values - 1: left, 2: right.
        "align": UnknownType,
        # baseline for text alignment of axis values - 1: inside, 2: outside.
        "alignTo": UnknownType,
        # gridlines to draw from this axis' splits.
        "grid": Axis_Grid,
        # ticks to draw from this axis' splits.
        "ticks": Axis_Ticks,
        # axis border/edge rendering.
        "border": Axis_Border,
    },
    total=False,
)

Cursor = TypedDict(
    "Cursor",
    {
        # cursor on/off.
        "show": bool,
        # vertical crosshair on/off.
        "x": bool,
        # horizontal crosshair on/off.
        "y": bool,
        # cursor position left offset in CSS pixels (relative to plotting area).
        "left": float,
        # cursor position top offset in CSS pixels (relative to plotting area).
        "top": float,
        # closest data index to cursor (closestIdx).
        "idx": float | None,
        # returns data idx used for hover points & legend display (defaults to closestIdx).
        "dataIdx": JSCallback,
        # a series-matched array of indices returned by dataIdx().
        "idxs": list[float | None],
        # fires on debounced mousemove events; returns refined [left, top] tuple to snap cursor position.
        "move": JSCallback,
        # series hover points.
        "points": Cursor_Points,
        # event listener proxies (can be overridden to tweak interaction behavior).
        "bind": Cursor_Bind,
        # determines vt/hz cursor dragging to set selection & setScale (zoom).
        "drag": Cursor_Drag,
        # sync cursor between multiple charts.
        "sync": Cursor_Sync,
        # focus series closest to cursor (y).
        "focus": Cursor_Focus,
        # hover data points closest to cursor (x).
        "hover": Cursor_Hover,
        # lock cursor on mouse click in plotting area.
        "lock": bool,
        # the most recent mouse event.
        "event": DOMElement,
    },
    total=False,
)

Options = TypedDict(
    "Options",
    {
        "series": Required[list[Series]],
        # 1: aligned & ordered, single-x / y-per-series, 2: unordered & faceted, per-series/per-point x,y,size,label,color,shape,etc.
        "mode": Mode,
        # chart title.
        "title": str,
        # id to set on chart div.
        "id": str,
        # className to add to chart div.
        "class": str,
        # initial devicePixelRatio, if different than window.devicePixelRatio.
        "pxRatio": float,
        # data for chart, if none is provided as argument to constructor.
        "data": UnknownType,
        # converts a unix timestamp to Date that's time-adjusted for the desired timezone.
        "tzDate": UnknownType,
        # creates an efficient formatter for Date objects from a template string, e.g. {YYYY}-{MM}-{DD}.
        "fmtDate": UnknownType,
        # timestamp multiplier that yields 1 millisecond.
        "ms": Literal[1],
        # drawing order for axes/grid & series (default: ["axes", "series"]).
        "drawOrder": list[UnknownType],
        # whether vt & hz lines of series/grid/ticks should be crisp/sharp or sub-px antialiased.
        "pxAlign": bool | float,
        "bands": list[Band],
        "scales": Scales,
        "axes": list[Axis],
        # padding per side, in CSS pixels (can prevent cross-axis labels at the plotting area limits from being chopped off).
        "padding": UnknownType,
        "select": Select,
        "legend": Legend,
        "cursor": Cursor,
        "focus": Focus,
        "hooks": UnknownType,
        "plugins": list[Plugin],
    },
    total=False,
)
