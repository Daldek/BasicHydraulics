# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BasicHydraulics is a Python toolkit for river channel and hydraulic structure capacity calculations.

Repository: https://github.com/Daldek/BasicHydraulics

It implements the Manning formula for open channel flow, Torricelli's law for flow through openings, and FHWA HY-8/HDS-5 methodology for culvert hydraulics.

## Installation

```bash
# Install in development mode
pip install -e .

# With GUI support
pip install -e ".[gui]"

# With development dependencies
pip install -e ".[dev]"
```

## Running Examples

```bash
jupyter notebook Examples.ipynb
```

## Running GUI

```bash
basichydraulics-gui
```

Opens web-based interface at http://localhost:8501.

## Project Structure

```
BasicHydraulics/
    basichydraulics/           # Main package
        __init__.py            # Package exports
        channel.py             # Open channel flow
        structure.py           # Flow through openings
        culvert.py             # Culvert hydraulics (FHWA HY-8)
        gui.py                 # Streamlit GUI
    tests/                     # Test suite
        test_culvert.py        # Culvert unit tests
    pyproject.toml             # Package configuration
    Examples.ipynb             # Usage examples
```

## Architecture

### basichydraulics/channel.py - Open Channel Flow

Base class `Channel` implements Manning formula calculations:
- `hydraulic_radius()` - cross-sectional area / wetted perimeter
- `velocity()` - uses Manning's n roughness coefficient
- `flow_rate()` - velocity × cross-sectional area

Subclasses define geometry-specific calculations for `cross_sectional_area()` and `wetted_perimeter()`:
- `Flat` - sheet flow over flat surfaces
- `TriangularChannel` - V-shaped channels (bank_slope parameter)
- `RectangularChannel` - box channels
- `TrapezoidalChannel` - sloped banks (bank_inclination_m parameter)
- `SemiCircularChannel` - partial pipe flow using circular segment geometry
- `IrregularChannel` - arbitrary cross-sections defined by station-elevation pairs, supports multiple independent wet segments
  - LOB/ROB subsections with `lob_station`, `rob_station` parameters
  - Separate roughness coefficients: `lob_n`, `rob_n` for overbanks
  - `get_subsections_summary()` - flow distribution by LOB/Channel/ROB

All channel classes have:
- `calculate_flow_rates()` - generates depth/water_level to flow rate mapping
- `get_cross_section()` - returns (stations, elevations) for visualization

### Visualization

`plot_channel(channel, title=None)` - creates side-by-side plots:
- Cross-section view with water level
- Flow rate curve with current point marked
- Synchronized Y axes between both plots

### basichydraulics/structure.py - Flow Through Openings

Base class `Structure` handles opening geometry:
- `dimension` accepts float (circular diameter) or list of 2 floats (rectangular width × height)
- `opening_area()` calculates πr²/4 for circular, w×h for rectangular
- `get_opening_profile()` returns dict with type, dimensions, and area

`SmallOpening` - Torricelli's law for orifice flow:
- Uses velocity_coef (default 0.98) and contraction_coef (default 0.64)
- `velocity()`, `flow()`, `time_to_discharge()`, `flow_duration_for_volume()`
- `calculate_flow_curves(head_min, head_max, step)` - generates Q(h) mapping starting from head=0.0

`LargeOpening` - accounts for varying head across opening height (Boussinesq formula):
- `free_flow()` - unsubmerged discharge
- `submerged_flow()` - fully underwater
- `partially_submerged_flow()` - mixed conditions
- `calculate_flow_curves(head_min, head_max, step, flow_type)` - generates Q(h) mapping
  - `flow_type`: 'free', 'submerged', or 'all' (returns both)

### Structure Visualization

`plot_structure(structure, head_max=None, title=None)` - creates flow rate curve plot:
- Flow rate Q vs hydraulic head h
- Current operating point marked in red
- Curve starts from (0, 0)

### basichydraulics/culvert.py - Culvert Hydraulics

Implements FHWA HY-8/HDS-5 methodology for culvert flow analysis with automatic inlet/outlet control regime selection.

Base class `Culvert` (ABC) provides:
- `inlet_control_flow()` - uses FHWA inlet control equations (unsubmerged/submerged)
- `outlet_control_flow()` - uses energy equation with entrance, friction, and exit losses
- `controlling_flow()` - returns (Q, regime) selecting the controlling regime (lower flow)
- `velocity_inlet()`, `velocity_outlet()` - velocities at culvert ends
- `calculate_rating_curve(hw_min, hw_max, step)` - generates headwater-discharge rating curve
  - `hw_min` defaults to `step` (starts from first positive headwater, not 0)

Geometry subclasses:
- `CircularCulvert` - pipe culverts (parameter: `diameter`)
- `BoxCulvert` - rectangular culverts (parameters: `width`, `height`)
- `PipeArchCulvert` - pipe-arch/elliptical culverts (parameters: `span_width`, `rise_height`)

Common parameters for all culvert types:
- `length` - culvert length [m]
- `slope` - culvert bed slope [m/m]
- `inlet_type` - inlet geometry ('projecting', 'headwall', 'mitered', 'wingwall_XX')
- `roughness_n` - Manning's roughness coefficient
- `headwater` - headwater depth above inlet invert [m]
- `tailwater` - tailwater depth above outlet invert [m]
- `Ke` - entrance loss coefficient (auto-selected if None)
- `downstream_channel` - optional Channel object for iterative tailwater calculation
- `outlet_invert_offset` - elevation difference between outlet invert and channel bottom [m]

Inlet coefficients from FHWA HDS-5 tables are stored in class constants:
- `INLET_COEFFICIENTS` - K, M, c for unsubmerged; c_sub, Y, Ks for submerged
- `ENTRANCE_LOSS` - Ke values for each inlet type

#### Downstream Channel Integration

When `downstream_channel` is provided, tailwater is calculated iteratively from the channel's rating curve instead of using a fixed value.

The `outlet_invert_offset` parameter accounts for elevation difference between:
- Culvert outlet invert
- Downstream channel bottom (lowest point)

```
offset > 0: Outlet above channel bottom → lower tailwater
offset = 0: Outlet at channel bottom (default)
offset < 0: Outlet below channel bottom → higher tailwater
```

Example with offset:
```python
from basichydraulics.channel import RectangularChannel, IrregularChannel
from basichydraulics.culvert import CircularCulvert

# Regular channel (bottom at 0)
channel = RectangularChannel(depth=2.0, width=5.0, inclination=0.001, roughness_n=0.03)
culvert = CircularCulvert(
    diameter=1.2, length=30, slope=0.01, headwater=2.0,
    downstream_channel=channel,
    outlet_invert_offset=0.3  # Outlet 0.3m above channel bottom
)
Q, regime = culvert.controlling_flow()
print(f"Tailwater = {culvert.tailwater:.3f} m")

# IrregularChannel with non-zero bottom elevation
irregular = IrregularChannel(
    stations=[0, 5, 10, 15, 20],
    elevations=[105, 102, 100, 102, 105],  # Bottom at 100.0 m
    water_level=103.0,
    inclination=0.001,
    roughness_n=0.03
)
culvert2 = CircularCulvert(
    diameter=1.2, length=30, slope=0.01, headwater=2.0,
    downstream_channel=irregular,
    outlet_invert_offset=0.5  # Outlet at 100.5 m (0.5m above channel bottom)
)
```

Helper methods for downstream channel:
- `_build_tailwater_curve()` - builds Q → depth lookup from channel rating curve
- `_interpolate_tailwater(Q)` - interpolates tailwater for given discharge, applies offset
- `tailwater_exceeded_capacity()` - returns True if Q exceeded channel capacity

### Culvert Visualization

`plot_culvert(culvert, hw_max=None, title=None)` - creates side-by-side plots:
- Left: Cross-section view with headwater level
- Right: Rating curve (Q vs HW) as single continuous line
- Current operating point marked with green star (shows regime in annotation)

### basichydraulics/gui.py - Web Interface

Streamlit-based GUI providing:

**Channel Calculator:**
- Channel type dropdown selector
- Dynamic parameter input fields (adapt to selected channel type)
- Two-column station/elevation data entry for IrregularChannel
- LOB/ROB boundary configuration with separate Manning's n coefficients
- Results table: cross-sectional area, wetted perimeter, hydraulic radius, velocity, flow rate
- Subsection analysis table (for IrregularChannel with LOB/ROB)
- Flow rate curve as interactive dataframe
- Visualization plot (cross-section + flow rate curve)

**Culvert Calculator:**
- Culvert type selector (Circular, Box, Pipe-Arch)
- Downstream channel integration:
  - Checkbox to use channel from above for tailwater
  - Outlet invert offset input (elevation difference from channel bottom)
- Parameter inputs: length, slope, Manning's n, headwater, tailwater
- Geometry-specific inputs: diameter, width/height, span/rise
- Inlet type selector with auto Ke selection
- Results: discharge, control regime, velocities, Ke, calculated tailwater (if using downstream channel)
- Warning displayed if discharge exceeds downstream channel capacity
- Rating curve table with regime indication
- Visualization (cross-section + rating curve with regime coloring)

## Key Constants

- Standard gravity `g = 9.80665` m/s² (structure.py)
- Default fluid density: 1000 kg/m³ (water)
