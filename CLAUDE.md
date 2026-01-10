# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BasicHydraulics is a Python toolkit for river channel and hydraulic structure capacity calculations.

Repository: https://github.com/Daldek/BasicHydraulics

It implements the Manning formula for open channel flow and Torricelli's law for flow through openings.

## Environment Setup

```bash
source venv/bin/activate
```

## Running Examples

```bash
jupyter notebook Examples.ipynb
```

## Architecture

### channel.py - Open Channel Flow

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

All channel classes have:
- `calculate_flow_rates()` - generates depth/water_level to flow rate mapping
- `get_cross_section()` - returns (stations, elevations) for visualization

### Visualization

`plot_channel(channel, title=None)` - creates side-by-side plots:
- Cross-section view with water level
- Flow rate curve with current point marked
- Synchronized Y axes between both plots

### structure.py - Flow Through Openings

Base class `Structure` handles opening geometry:
- `dimension` accepts float (circular diameter) or list of 2 floats (rectangular width × height)
- `opening_area()` calculates πr²/4 for circular, w×h for rectangular

`SmallOpening` - Torricelli's law for orifice flow:
- Uses velocity_coef (default 0.98) and contraction_coef (default 0.64)
- `velocity()`, `flow()`, `time_to_discharge()`, `flow_duration_for_volume()`

`LargeOpening` - accounts for varying head across opening height:
- `free_flow()` - unsubmerged discharge
- `submerged_flow()` - fully underwater
- `partially_submerged_flow()` - mixed conditions

## Key Constants

- Standard gravity `g = 9.80665` m/s² (structure.py)
- Default fluid density: 1000 kg/m³ (water)
