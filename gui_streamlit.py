#!/usr/bin/env python3
"""
BasicHydraulics GUI Application (Streamlit version)

A web-based graphical interface for performing hydraulic channel calculations.

Features:
    - Channel type selector (Rectangular, Trapezoidal, Triangular, Semi-Circular, Flat, Irregular)
    - Dynamic input fields based on selected channel type
    - Two-column station/elevation input for irregular channels
    - LOB/ROB (Left/Right Overbank) subsection configuration with separate Manning's n
    - Real-time calculation of hydraulic parameters:
        - Cross-sectional area, wetted perimeter, hydraulic radius
        - Flow velocity and discharge (flow rate)
        - Subsection flow distribution for irregular channels
    - Flow rate curve table
    - Interactive visualization (cross-section + flow rate curve)

Run with:
    streamlit run gui_streamlit.py

Opens in browser at http://localhost:8501
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from channel import (
    Flat, TriangularChannel, RectangularChannel,
    TrapezoidalChannel, SemiCircularChannel, IrregularChannel,
    plot_channel
)

# Page configuration
st.set_page_config(
    page_title="BasicHydraulics",
    page_icon="🌊",
    layout="wide"
)

st.title("BasicHydraulics - Channel Flow Calculator")

# Channel type selector
CHANNEL_TYPES = [
    "Select channel type...",
    "Rectangular",
    "Trapezoidal",
    "Triangular",
    "Semi-Circular",
    "Flat",
    "Irregular"
]

channel_type = st.selectbox("Channel Type", CHANNEL_TYPES)

# Initialize channel variable
channel = None

if channel_type != "Select channel type...":

    # Create two columns for input and results
    col_input, col_results = st.columns([1, 1.5])

    with col_input:
        st.subheader("Parameters")

        # Common parameters for all channel types
        inclination = st.number_input("Inclination (slope)", value=0.001, format="%.6f", step=0.0001)
        roughness_n = st.number_input("Manning's n", value=0.03, format="%.4f", step=0.001)

        # Channel-specific parameters
        if channel_type == "Rectangular":
            depth = st.number_input("Depth [m]", value=2.0, min_value=0.01, step=0.1)
            width = st.number_input("Width [m]", value=5.0, min_value=0.01, step=0.1)

            channel = RectangularChannel(
                depth=depth,
                width=width,
                inclination=inclination,
                roughness_n=roughness_n
            )

        elif channel_type == "Trapezoidal":
            depth = st.number_input("Depth [m]", value=1.5, min_value=0.01, step=0.1)
            width = st.number_input("Bottom width [m]", value=3.0, min_value=0.01, step=0.1)
            bank_inclination_m = st.number_input("Bank slope (H:V)", value=1.5, min_value=0.01, step=0.1)

            channel = TrapezoidalChannel(
                depth=depth,
                width=width,
                bank_inclination_m=bank_inclination_m,
                inclination=inclination,
                roughness_n=roughness_n
            )

        elif channel_type == "Triangular":
            depth = st.number_input("Depth [m]", value=2.0, min_value=0.01, step=0.1)
            bank_slope = st.number_input("Bank slope (H:V)", value=1.0, min_value=0.01, step=0.1)

            channel = TriangularChannel(
                depth=depth,
                bank_slope=bank_slope,
                inclination=inclination,
                roughness_n=roughness_n
            )

        elif channel_type == "Semi-Circular":
            depth = st.number_input("Water depth [m]", value=0.8, min_value=0.01, step=0.1)
            radius = st.number_input("Pipe radius [m]", value=1.0, min_value=0.01, step=0.1)

            if depth > 2 * radius:
                st.error(f"Water depth cannot exceed pipe diameter ({2*radius:.2f} m)")
            else:
                channel = SemiCircularChannel(
                    depth=depth,
                    radius=radius,
                    inclination=inclination,
                    roughness_n=roughness_n
                )

        elif channel_type == "Flat":
            depth = st.number_input("Depth [m]", value=0.5, min_value=0.01, step=0.1)
            width = st.number_input("Width [m]", value=10.0, min_value=0.01, step=0.1)

            channel = Flat(
                depth=depth,
                width=width,
                inclination=inclination,
                roughness_n=roughness_n
            )

        elif channel_type == "Irregular":
            water_level = st.number_input("Water level [m]", value=5.0, step=0.1)

            st.markdown("---")
            st.markdown("**Overbank boundaries (optional)**")

            col_lob, col_rob = st.columns(2)
            with col_lob:
                lob_station = st.number_input("LOB station", value=None, format="%.2f",
                                              help="Left Overbank boundary station")
                lob_n = st.number_input("LOB Manning's n", value=None, format="%.4f",
                                        help="Manning's n for left overbank")
            with col_rob:
                rob_station = st.number_input("ROB station", value=None, format="%.2f",
                                              help="Right Overbank boundary station")
                rob_n = st.number_input("ROB Manning's n", value=None, format="%.4f",
                                        help="Manning's n for right overbank")

            st.markdown("---")
            st.markdown("**Station-Elevation Data**")
            st.caption("Enter values in each column (one per line)")

            default_stations = """0
1
2
3
4
5
6
7
8
9
10"""
            default_elevations = """6
5
5
4
2
1
2
4
5
5
6"""

            col_sta, col_elev = st.columns(2)
            with col_sta:
                stations_text = st.text_area(
                    "Stations [m]",
                    value=default_stations,
                    height=200
                )
            with col_elev:
                elevations_text = st.text_area(
                    "Elevations [m]",
                    value=default_elevations,
                    height=200
                )

            # Parse station-elevation data
            stations = []
            elevations = []
            parse_error = None

            try:
                stations = [float(x.strip()) for x in stations_text.strip().split('\n') if x.strip()]
                elevations = [float(x.strip()) for x in elevations_text.strip().split('\n') if x.strip()]
            except ValueError as e:
                parse_error = f"Invalid number format: {e}"

            if parse_error:
                st.error(parse_error)
            elif len(stations) != len(elevations):
                st.error(f"Number of stations ({len(stations)}) must match number of elevations ({len(elevations)})")
            elif len(stations) < 2:
                st.error("Need at least 2 station-elevation pairs")
            else:
                try:
                    channel = IrregularChannel(
                        stations=stations,
                        elevations=elevations,
                        water_level=water_level,
                        inclination=inclination,
                        roughness_n=roughness_n,
                        lob_station=lob_station if lob_station else None,
                        rob_station=rob_station if rob_station else None,
                        lob_n=lob_n if lob_n else None,
                        rob_n=rob_n if rob_n else None
                    )
                except Exception as e:
                    st.error(f"Error creating channel: {e}")

    # Results column
    with col_results:
        if channel is not None:
            st.subheader("Results")

            # Basic results
            results_data = {
                "Parameter": [
                    "Cross-sectional Area",
                    "Wetted Perimeter",
                    "Hydraulic Radius",
                    "Velocity",
                    "Flow Rate"
                ],
                "Value": [
                    f"{channel.cross_sectional_area():.4f} m²",
                    f"{channel.wetted_perimeter():.4f} m",
                    f"{channel.hydraulic_radius():.4f} m",
                    f"{channel.velocity():.4f} m/s",
                    f"{channel.flow_rate():.4f} m³/s"
                ]
            }
            st.table(pd.DataFrame(results_data))

            # Subsection analysis for IrregularChannel with LOB/ROB
            if isinstance(channel, IrregularChannel):
                if channel.lob_station is not None or channel.rob_station is not None:
                    st.markdown("**Subsection Analysis**")
                    summary = channel.get_subsections_summary()

                    subsection_data = []
                    for name, data in summary.items():
                        if data['area'] > 0:
                            subsection_data.append({
                                "Subsection": name,
                                "Area [m²]": f"{data['area']:.2f}",
                                "Area %": f"{data['area_pct']}%",
                                "Flow [m³/s]": f"{data['flow']:.3f}",
                                "Flow %": f"{data['flow_pct']}%"
                            })

                    if subsection_data:
                        st.table(pd.DataFrame(subsection_data))

            # Flow rate curve
            st.markdown("**Flow Rate Curve**")
            flow_rates = channel.calculate_flow_rates()
            flow_df = pd.DataFrame([
                {"Depth/Level [m]": depth, "Flow Rate [m³/s]": flow}
                for depth, flow in flow_rates.items()
            ])
            st.dataframe(flow_df, width="stretch", hide_index=True)

    # Visualization
    if channel is not None:
        st.markdown("---")
        st.subheader("Visualization")

        try:
            fig = plot_channel(channel, channel_type)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Error creating plot: {e}")

else:
    st.info("Select a channel type to begin calculations")

# Footer
st.markdown("---")
st.caption("BasicHydraulics - Toolkit for river channel and structures capacity calculations")
