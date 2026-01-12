# BasicHydraulics

Toolkit for carrying out basic river channel and structures capacity calculations

## Features

- **Channel flow calculations** using Manning formula:
  - Flat, Triangular, Rectangular, Trapezoidal, Semi-Circular channels
  - Irregular cross-sections with multi-segment wet area support
  - Flow rate curves Q(h) with `calculate_flow_rates()`
  - Visualization with `plot_channel()` (cross-section + flow rate curve)
- **Structure flow calculations** using Torricelli's law:
  - Small openings (orifice flow)
  - Large openings (free, submerged, partially submerged)
  - Flow rate curves Q(h) with `calculate_flow_curves()`
  - Visualization with `plot_structure()` (flow rate curve)
- **Web GUI** for interactive calculations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Python API

```python
# Channel calculations
from channel import RectangularChannel, plot_channel

channel = RectangularChannel(depth=2, width=5, inclination=0.001, roughness_n=0.03)
channel.calculate_flow_rates()
plot_channel(channel, 'Rectangular Channel')

# Structure calculations
from structure import SmallOpening, LargeOpening, plot_structure

opening = SmallOpening(dimension=0.5, hydraulic_head=2.0)
opening.calculate_flow_curves(head_max=5.0)
plot_structure(opening, title='Small Opening Flow Curve')

large = LargeOpening(dimension=[1.0, 0.8], hydraulic_head=3.0)
large.calculate_flow_curves(head_max=5.0, flow_type='all')  # free + submerged
```

### Web GUI

```bash
streamlit run gui_streamlit.py
```

Opens in browser at http://localhost:8501 with:
- Channel type selector (Rectangular, Trapezoidal, Triangular, Semi-Circular, Flat, Irregular)
- Dynamic parameter input fields
- Two-column station/elevation input for irregular cross-sections
- LOB/ROB (Left/Right Overbank) subsection configuration
- Real-time results and visualization

## TODO list
- ~~Development of methods to determine the flow rate curve for channel classes where this functionality does not yet exist~~ ✓
- ~~Graphical representation of results~~ ✓
- ~~Web GUI for interactive calculations~~ ✓
- Development of flow through structures
- Saving results to file
