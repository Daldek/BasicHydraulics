# BasicHydraulics

Toolkit for carrying out basic river channel and structures capacity calculations

## Features

- **Channel flow calculations** using Manning formula:
  - Flat, Triangular, Rectangular, Trapezoidal, Semi-Circular channels
  - Irregular cross-sections with multi-segment wet area support
- **Structure flow calculations** using Torricelli's law:
  - Small openings (orifice flow)
  - Large openings (free, submerged, partially submerged)
- **Visualization** with `plot_channel()` function (cross-section + flow rate curve)

## Usage

```python
from channel import RectangularChannel, plot_channel

channel = RectangularChannel(depth=2, width=5, inclination=0.001, roughness_n=0.03)
channel.calculate_flow_rates()
plot_channel(channel, 'Rectangular Channel')
```

## TODO list
- ~~Development of methods to determine the flow rate curve for channel classes where this functionality does not yet exist~~ ✓
- Development of flow through structures
- Saving results to file
- ~~Graphical representation of results~~ ✓
