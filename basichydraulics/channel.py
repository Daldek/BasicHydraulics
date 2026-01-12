import math
from fractions import Fraction

"""
https://en.wikipedia.org/wiki/Manning_formula
"""


class Channel:
    """Simple class to define channel's XS"""

    def __init__(self, inclination, roughness_n):
        """Basic input parameters"""
        self.inclination = inclination
        self.roughness_n = roughness_n
        self._hydraulic_radius = None
        self._velocity = None
        self._flow_rate = None

    def hydraulic_radius(self):
        """
        https://en.wikipedia.org/wiki/Manning_formula#Hydraulic_radius
        """
        if self._hydraulic_radius is None:
            self._hydraulic_radius = (
                self.cross_sectional_area() / self.wetted_perimeter()
            )
            # print(f"Hydraulic radius is {self._hydraulic_radius:.2f} m")
        return self._hydraulic_radius

    def velocity(self):
        if self._velocity is None:
            self._velocity = (
                (1 / self.roughness_n)
                * math.pow(self.hydraulic_radius(), Fraction(2, 3))
                * math.pow(self.inclination, 0.5)
            )
            # print(f"Velocity is {self._velocity:.2f} meters per second")
        return self._velocity

    def flow_rate(self):
        if self._flow_rate is None:
            self._flow_rate = self.velocity() * self.cross_sectional_area()
            # print(f"Max flow is: {self._flow_rate:.3f} cubic meters per second")
        return self._flow_rate


class Flat(Channel):
    """
    Class for flat surfaces
    """

    def __init__(self, depth, width, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.width = width
        self._cross_sectional_area = None
        self._wetted_perimeter = None

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = self.depth * self.width
        return self._cross_sectional_area

    def wetted_perimeter(self):
        if self._wetted_perimeter is None:
            self._wetted_perimeter = self.width
        return self._wetted_perimeter

    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, int((self.depth * 100) + 1), 10):
            self.depth = h / 100.0
            self._cross_sectional_area = None
            self._wetted_perimeter = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            flow_rates[self.depth] = round(self.flow_rate(), 2)
        return flow_rates

    def get_cross_section(self):
        """
        Return station-elevation pairs for cross-section visualization.
        Flat surface: horizontal line at elevation 0.
        """
        stations = [0, self.width]
        elevations = [0, 0]
        return stations, elevations


class TriangularChannel(Channel):
    """Class for triangular channels"""

    def __init__(self, depth, bank_slope, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.bank_slope = bank_slope
        self._cross_sectional_area = None
        self._wetted_perimeter = None
        self._base_width = None

    def base_width(self):
        """
        Calculated for half of the cross-section, i.e. a right-angled triangle.
        """
        if self._base_width is None:
            self._base_width = self.depth * self.bank_slope
            print(f"Base width is {self._base_width:.2f} m")
        return self._base_width

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = self.base_width() * self.depth
            print(f"XS area is {self._cross_sectional_area:.2f} sqm")
        return self._cross_sectional_area

    def wetted_perimeter(self):
        """
        https://en.wikipedia.org/wiki/Wetted_perimeter
        """
        if self._wetted_perimeter is None:
            self._wetted_perimeter = 2 * math.sqrt(
                math.pow(self.depth, 2) + math.pow(self.base_width(), 2)
            )
            print(f"Wetter perimeter is {self._wetted_perimeter:.2f} m")
        return self._wetted_perimeter

    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, int((self.depth * 100) + 1), 10):
            self.depth = h / 100.0
            self._base_width = None
            self._cross_sectional_area = None
            self._wetted_perimeter = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            if self.depth > 0:
                flow_rates[self.depth] = round(self.flow_rate(), 2)
            else:
                flow_rates[self.depth] = 0.0
        return flow_rates

    def get_cross_section(self):
        """
        Return station-elevation pairs for cross-section visualization.
        Triangular channel: V-shape with bottom at center.
        """
        half_width = self.depth * self.bank_slope
        stations = [0, half_width, 2 * half_width]
        elevations = [self.depth, 0, self.depth]
        return stations, elevations


class RectangularChannel(Channel):
    """Class for rectangular channels"""

    def __init__(self, depth, width, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.width = width
        self._cross_sectional_area = None
        self._wetted_perimeter = None

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = self.width * self.depth
        return self._cross_sectional_area

    def wetted_perimeter(self):
        """
        https://en.wikipedia.org/wiki/Wetted_perimeter
        """
        if self._wetted_perimeter is None:
            self._wetted_perimeter = 2 * self.depth + self.width
        return self._wetted_perimeter

    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, int((self.depth * 100) + 1), 10):
            self.depth = h / 100.0
            self._cross_sectional_area = None
            self._wetted_perimeter = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            flow_rates[self.depth] = round(self.flow_rate(), 2)
        return flow_rates

    def get_cross_section(self):
        """
        Return station-elevation pairs for cross-section visualization.
        Rectangular channel: vertical walls.
        """
        stations = [0, 0, self.width, self.width]
        elevations = [self.depth, 0, 0, self.depth]
        return stations, elevations


class TrapezoidalChannel(Channel):
    """Class for trapezoidal channels"""

    def __init__(self, depth, width, bank_inclination_m, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.width = width
        self.bank_inclination_m = bank_inclination_m
        self._cross_sectional_area = None
        self._wetted_perimeter = None

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = (
                (self.depth * self.bank_inclination_m) + self.width
            ) * self.depth
        return self._cross_sectional_area

    def wetted_perimeter(self):
        """
        https://en.wikipedia.org/wiki/Wetted_perimeter
        """
        if self._wetted_perimeter is None:
            self._wetted_perimeter = self.width + 2 * self.depth * math.sqrt(
                1 + math.pow(self.bank_inclination_m, 2)
            )
        return self._wetted_perimeter

    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, int((self.depth * 100) + 1), 10):
            self.depth = h / 100.0
            self._cross_sectional_area = None
            self._wetted_perimeter = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            flow_rates[self.depth] = round(self.flow_rate(), 2)
        return flow_rates

    def get_cross_section(self):
        """
        Return station-elevation pairs for cross-section visualization.
        Trapezoidal channel: sloped banks with flat bottom.
        """
        bank_offset = self.depth * self.bank_inclination_m
        stations = [0, bank_offset, bank_offset + self.width, 2 * bank_offset + self.width]
        elevations = [self.depth, 0, 0, self.depth]
        return stations, elevations


class SemiCircularChannel(Channel):
    """Class for circular channels"""

    def __init__(self, depth, radius, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.radius = radius
        self._theta_rad = None
        self._cross_sectional_area = None
        self._wetted_perimeter = None

    def theta_rad(self):
        """
        https://en.wikipedia.org/wiki/Radian
        https://en.wikipedia.org/wiki/Sine_and_cosine
        The theta angle was calculated as the cosine of the ratio of the height of the right triangle
        (radius of circle - depth of water (sagitta)), i.e. the perpendicular lying at the acute angle,
        to the radius, i.e. the opposite perpendicular.
        The result is then doubled to obtain the angle theta for an isosceles triangle
        """
        if self._theta_rad is None:
            self._theta_rad = 2 * math.acos((self.radius - self.depth) / self.radius)
        return self._theta_rad

    def cross_sectional_area(self):
        """
        https://en.wikipedia.org/wiki/Circular_segment
        """
        if self._cross_sectional_area is None:
            self._cross_sectional_area = (
                0.5
                * math.pow(self.radius, 2)
                * (self.theta_rad() - math.sin(self.theta_rad()))
            )
        return self._cross_sectional_area

    def wetted_perimeter(self):
        """
        https://en.wikipedia.org/wiki/Wetted_perimeter
        """
        if self._wetted_perimeter is None:
            self._wetted_perimeter = self.radius * self.theta_rad()
        return self._wetted_perimeter

    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, int((self.depth * 100) + 1), 10):
            self.depth = h / 100.0
            self._theta_rad = None
            self._cross_sectional_area = None
            self._wetted_perimeter = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            if self.depth > 0:  # theta_rad requires depth > 0
                flow_rates[self.depth] = round(self.flow_rate(), 2)
            else:
                flow_rates[self.depth] = 0.0
        return flow_rates

    def get_cross_section(self, num_points=50):
        """
        Return station-elevation pairs for cross-section visualization.
        Semi-circular channel: arc of a circle.

        Args:
            num_points (int): Number of points to approximate the arc.
        """
        stations = []
        elevations = []
        # Generate points along the circular arc (bottom half of circle)
        for i in range(num_points + 1):
            angle = math.pi + (math.pi * i / num_points)  # from pi to 2*pi
            x = self.radius + self.radius * math.cos(angle)
            y = self.radius + self.radius * math.sin(angle)
            stations.append(x)
            elevations.append(y)
        return stations, elevations


class IrregularChannel(Channel):
    """
    Class for irregular cross-sections defined by station-elevation pairs.

    Station represents horizontal distance, elevation represents vertical height.
    Water level is specified as an absolute elevation.

    Supports discontinuous wet areas - when the cross-section has multiple
    independent wet segments, each segment's flow is calculated separately
    using its own hydraulic radius, then summed for total flow.

    Optionally supports subdivision into Left Overbank (LOB), Main Channel,
    and Right Overbank (ROB) with separate roughness coefficients for each.
    """

    def __init__(self, stations, elevations, water_level, inclination, roughness_n,
                 lob_station=None, rob_station=None, lob_n=None, rob_n=None):
        """
        Initialize an IrregularChannel instance.

        Args:
            stations (list): List of horizontal distances (x-coordinates)
            elevations (list): List of elevations (y-coordinates) corresponding to stations
            water_level (float): Absolute elevation of water surface
            inclination (float): Channel bed slope
            roughness_n (float): Manning's roughness coefficient for main channel
            lob_station (float, optional): Station marking left overbank boundary
            rob_station (float, optional): Station marking right overbank boundary
            lob_n (float, optional): Manning's n for left overbank (defaults to roughness_n)
            rob_n (float, optional): Manning's n for right overbank (defaults to roughness_n)
        """
        super().__init__(inclination, roughness_n)
        if len(stations) != len(elevations):
            raise ValueError("stations and elevations must have the same length")
        if len(stations) < 2:
            raise ValueError("At least 2 points are required to define a cross-section")
        self.stations = stations
        self.elevations = elevations
        self.water_level = water_level
        self.lob_station = lob_station
        self.rob_station = rob_station
        self.lob_n = lob_n if lob_n is not None else roughness_n
        self.rob_n = rob_n if rob_n is not None else roughness_n
        self._cross_sectional_area = None
        self._wetted_perimeter = None
        self._wet_segments = None
        self._subsection_flows = None

    def _interpolate_station(self, s1, e1, s2, e2, water_level):
        """Interpolate station at water level between two points."""
        if e1 == e2:
            return s1
        return s1 + (s2 - s1) * (water_level - e1) / (e2 - e1)

    def _find_wet_segments(self):
        """
        Find all independent wet segments in the cross-section.

        Returns:
            list: List of segments, each segment is a list of (station, elevation) tuples
        """
        if self._wet_segments is not None:
            return self._wet_segments

        segments = []
        current_segment = []
        in_water = False

        for i in range(len(self.stations)):
            s, e = self.stations[i], self.elevations[i]
            point_underwater = e <= self.water_level

            if i == 0:
                if point_underwater:
                    in_water = True
                    current_segment.append((s, e))
            else:
                s_prev, e_prev = self.stations[i - 1], self.elevations[i - 1]
                prev_underwater = e_prev <= self.water_level

                if not prev_underwater and point_underwater:
                    # Entering water - interpolate entry point
                    s_int = self._interpolate_station(s_prev, e_prev, s, e, self.water_level)
                    current_segment = [(s_int, self.water_level), (s, e)]
                    in_water = True
                elif prev_underwater and not point_underwater:
                    # Leaving water - interpolate exit point
                    s_int = self._interpolate_station(s_prev, e_prev, s, e, self.water_level)
                    current_segment.append((s_int, self.water_level))
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = []
                    in_water = False
                elif in_water:
                    # Still in water
                    current_segment.append((s, e))

        # Close last segment if still in water
        if in_water and len(current_segment) >= 2:
            segments.append(current_segment)

        self._wet_segments = segments
        return segments

    def _segment_area(self, segment):
        """Calculate cross-sectional area for a single wet segment."""
        area = 0.0
        for i in range(len(segment) - 1):
            s1, e1 = segment[i]
            s2, e2 = segment[i + 1]
            h1 = self.water_level - e1
            h2 = self.water_level - e2
            area += 0.5 * (h1 + h2) * (s2 - s1)
        return area

    def _segment_perimeter(self, segment):
        """Calculate wetted perimeter for a single wet segment."""
        perimeter = 0.0
        for i in range(len(segment) - 1):
            s1, e1 = segment[i]
            s2, e2 = segment[i + 1]
            perimeter += math.sqrt((s2 - s1) ** 2 + (e2 - e1) ** 2)
        return perimeter

    def _interpolate_elevation(self, s1, e1, s2, e2, target_station):
        """Interpolate elevation at a given station between two points."""
        if s1 == s2:
            return e1
        return e1 + (e2 - e1) * (target_station - s1) / (s2 - s1)

    def _split_segment_by_boundaries(self, segment):
        """
        Split a wet segment at LOB and ROB boundaries.

        Returns:
            list: List of (subsection_name, sub_segment) tuples
        """
        if self.lob_station is None and self.rob_station is None:
            return [('Channel', segment)]

        # Collect all boundary stations that fall within this segment
        seg_stations = [p[0] for p in segment]
        seg_min, seg_max = min(seg_stations), max(seg_stations)

        boundaries = []
        if self.lob_station is not None and seg_min < self.lob_station < seg_max:
            boundaries.append(self.lob_station)
        if self.rob_station is not None and seg_min < self.rob_station < seg_max:
            boundaries.append(self.rob_station)

        if not boundaries:
            # No boundaries within segment - assign based on center
            center = (seg_min + seg_max) / 2
            if self.lob_station is not None and center < self.lob_station:
                return [('LOB', segment)]
            elif self.rob_station is not None and center > self.rob_station:
                return [('ROB', segment)]
            else:
                return [('Channel', segment)]

        # Split segment at boundaries
        boundaries = sorted(boundaries)
        result = []
        current_segment = []
        boundary_idx = 0

        for i, (s, e) in enumerate(segment):
            # Check if we need to split before this point
            while boundary_idx < len(boundaries) and boundaries[boundary_idx] <= s:
                if current_segment:
                    # Interpolate point at boundary
                    if i > 0:
                        s_prev, e_prev = segment[i - 1]
                        e_at_boundary = self._interpolate_elevation(
                            s_prev, e_prev, s, e, boundaries[boundary_idx])
                        current_segment.append((boundaries[boundary_idx], e_at_boundary))

                    # Determine subsection for current_segment
                    seg_center = sum(p[0] for p in current_segment) / len(current_segment)
                    subsection = self._get_subsection_by_station(seg_center)
                    result.append((subsection, current_segment))

                    # Start new segment from boundary
                    current_segment = [(boundaries[boundary_idx], e_at_boundary)]

                boundary_idx += 1

            current_segment.append((s, e))

        # Add final segment
        if len(current_segment) >= 2:
            seg_center = sum(p[0] for p in current_segment) / len(current_segment)
            subsection = self._get_subsection_by_station(seg_center)
            result.append((subsection, current_segment))

        return result

    def _get_subsection_by_station(self, station):
        """Determine subsection based on station value."""
        if self.lob_station is not None and station < self.lob_station:
            return 'LOB'
        elif self.rob_station is not None and station > self.rob_station:
            return 'ROB'
        else:
            return 'Channel'

    def _get_roughness_for_subsection(self, subsection):
        """Get Manning's n for a subsection name."""
        if subsection == 'LOB':
            return self.lob_n
        elif subsection == 'ROB':
            return self.rob_n
        else:
            return self.roughness_n

    def _subsegment_flow(self, subsegment, roughness_n):
        """
        Calculate flow rate for a subsegment using Manning's equation.
        """
        area = self._segment_area(subsegment)
        perimeter = self._segment_perimeter(subsegment)

        if perimeter == 0 or area == 0:
            return 0.0

        hydraulic_radius = area / perimeter
        velocity = (
            (1 / roughness_n)
            * math.pow(hydraulic_radius, Fraction(2, 3))
            * math.pow(self.inclination, 0.5)
        )
        return velocity * area

    def _segment_flow(self, segment):
        """
        Calculate flow rate for a wet segment, splitting by LOB/Channel/ROB boundaries.
        """
        subsegments = self._split_segment_by_boundaries(segment)
        total_flow = 0.0
        for subsection, subseg in subsegments:
            n = self._get_roughness_for_subsection(subsection)
            total_flow += self._subsegment_flow(subseg, n)
        return total_flow

    def cross_sectional_area(self):
        """
        Calculate total cross-sectional area below water level (sum of all segments).
        """
        if self._cross_sectional_area is None:
            segments = self._find_wet_segments()
            self._cross_sectional_area = sum(self._segment_area(seg) for seg in segments)
        return self._cross_sectional_area

    def wetted_perimeter(self):
        """
        Calculate total wetted perimeter (sum of all segments).
        """
        if self._wetted_perimeter is None:
            segments = self._find_wet_segments()
            self._wetted_perimeter = sum(self._segment_perimeter(seg) for seg in segments)
        return self._wetted_perimeter

    def flow_rate(self):
        """
        Calculate total flow rate as sum of flows from all independent wet segments.
        Each segment uses its own hydraulic radius for correct Manning calculation.
        """
        if self._flow_rate is None:
            segments = self._find_wet_segments()
            self._flow_rate = sum(self._segment_flow(seg) for seg in segments)
        return self._flow_rate

    def get_segments_info(self):
        """
        Get detailed information about each wet segment and its subsections.

        Returns:
            list: List of dicts with area, perimeter, hydraulic_radius, flow,
                  subsection, and roughness_n for each subsegment
        """
        segments = self._find_wet_segments()
        info = []
        idx = 1

        for seg in segments:
            subsegments = self._split_segment_by_boundaries(seg)
            for subsection, subseg in subsegments:
                area = self._segment_area(subseg)
                perimeter = self._segment_perimeter(subseg)
                n = self._get_roughness_for_subsection(subsection)

                if perimeter > 0:
                    r_h = area / perimeter
                    velocity = (
                        (1 / n)
                        * math.pow(r_h, Fraction(2, 3))
                        * math.pow(self.inclination, 0.5)
                    )
                    flow = velocity * area
                else:
                    r_h = 0
                    velocity = 0
                    flow = 0

                info.append({
                    "segment": idx,
                    "subsection": subsection,
                    "roughness_n": n,
                    "points": subseg,
                    "area": round(area, 4),
                    "perimeter": round(perimeter, 4),
                    "hydraulic_radius": round(r_h, 4),
                    "velocity": round(velocity, 4),
                    "flow": round(flow, 4),
                })
                idx += 1
        return info

    def get_subsections_summary(self):
        """
        Get flow summary by subsection (LOB, Channel, ROB).

        Returns:
            dict: Summary with area, flow, and percentage for each subsection
        """
        segments = self._find_wet_segments()
        summary = {'LOB': {'area': 0, 'flow': 0},
                   'Channel': {'area': 0, 'flow': 0},
                   'ROB': {'area': 0, 'flow': 0}}

        for seg in segments:
            subsegments = self._split_segment_by_boundaries(seg)
            for subsection, subseg in subsegments:
                area = self._segment_area(subseg)
                n = self._get_roughness_for_subsection(subsection)
                flow = self._subsegment_flow(subseg, n)
                summary[subsection]['area'] += area
                summary[subsection]['flow'] += flow

        total_flow = sum(s['flow'] for s in summary.values())
        total_area = sum(s['area'] for s in summary.values())

        for key in summary:
            summary[key]['area'] = round(summary[key]['area'], 4)
            summary[key]['flow'] = round(summary[key]['flow'], 4)
            summary[key]['flow_pct'] = round(100 * summary[key]['flow'] / total_flow, 1) if total_flow > 0 else 0
            summary[key]['area_pct'] = round(100 * summary[key]['area'] / total_area, 1) if total_area > 0 else 0

        return summary

    def calculate_flow_rates(self, step=0.1):
        """
        Calculate flow rates for different water levels.

        Args:
            step (float): Water level increment in meters (default 0.1m)

        Returns:
            dict: Water levels mapped to flow rates
        """
        flow_rates = {}
        min_elev = min(self.elevations)
        max_elev = max(self.elevations)

        water_level = min_elev
        while water_level <= max_elev:
            self.water_level = round(water_level, 2)
            self._cross_sectional_area = None
            self._wetted_perimeter = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            self._wet_segments = None

            if self.water_level > min_elev:
                flow_rates[self.water_level] = round(self.flow_rate(), 2)
            else:
                flow_rates[self.water_level] = 0.0

            water_level += step

        return flow_rates

    def get_cross_section(self):
        """
        Return station-elevation pairs for cross-section visualization.
        Returns the original stations and elevations.
        """
        return self.stations, self.elevations


def plot_channel(channel, title=None):
    """
    Visualize channel cross-section and flow rate curve side by side.

    Args:
        channel: Channel object (any subclass of Channel)
        title: Optional title for the plot

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Get cross-section data
    stations, elevations = channel.get_cross_section()

    # Determine water level and depth info
    if hasattr(channel, 'water_level'):
        water_level = channel.water_level
        min_elev = min(elevations)
        max_elev = max(elevations)
    else:
        water_level = channel.depth
        min_elev = 0
        max_elev = channel.depth

    # Plot 1: Cross-section
    ax1.fill_between(stations, elevations, max(elevations) + 0.5,
                     color='saddlebrown', alpha=0.6, label='Channel bed')
    ax1.plot(stations, elevations, 'k-', linewidth=2)

    # Water level line
    water_stations = [min(stations), max(stations)]
    water_elevations = [water_level, water_level] if hasattr(channel, 'water_level') else [channel.depth, channel.depth]

    # Fill water area
    water_fill_stations = stations.copy() if isinstance(stations, list) else list(stations)
    water_fill_elevations = elevations.copy() if isinstance(elevations, list) else list(elevations)

    if hasattr(channel, 'water_level'):
        # For IrregularChannel - fill below water level
        ax1.fill_between(water_fill_stations, water_fill_elevations,
                         [min(water_level, e) for e in water_fill_elevations],
                         where=[e <= water_level for e in water_fill_elevations],
                         color='steelblue', alpha=0.5, label='Water')
        ax1.axhline(y=water_level, color='blue', linestyle='--', linewidth=1.5, label=f'Water level: {water_level:.2f} m')

        # Draw LOB/ROB boundaries if defined
        if hasattr(channel, 'lob_station') and channel.lob_station is not None:
            ax1.axvline(x=channel.lob_station, color='green', linestyle=':', linewidth=2, label=f'LOB: {channel.lob_station}')
        if hasattr(channel, 'rob_station') and channel.rob_station is not None:
            ax1.axvline(x=channel.rob_station, color='orange', linestyle=':', linewidth=2, label=f'ROB: {channel.rob_station}')
    else:
        # For other channels - fill from bottom to depth
        ax1.axhline(y=water_level, color='blue', linestyle='--', linewidth=1.5, label=f'Depth: {water_level:.2f} m')

    ax1.set_xlabel('Station [m]')
    ax1.set_ylabel('Elevation [m]')
    ax1.set_title('Cross-section')
    ax1.legend(loc='upper right', fontsize='small')

    # Plot 2: Flow rate curve
    flow_rates = channel.calculate_flow_rates()
    depths = list(flow_rates.keys())
    flows = list(flow_rates.values())

    sns.lineplot(x=flows, y=depths, ax=ax2, marker='o', markersize=4)
    ax2.set_xlabel('Flow rate [m³/s]')

    if hasattr(channel, 'water_level'):
        ax2.set_ylabel('Water level [m]')
        # Mark current water level
        current_flow = channel.flow_rate()
        ax2.scatter([current_flow], [channel.water_level], color='red', s=100, zorder=5)
        ax2.annotate(f'({current_flow:.2f}, {channel.water_level:.2f})',
                     (current_flow, channel.water_level),
                     textcoords="offset points", xytext=(10, 10), color='red')
    else:
        ax2.set_ylabel('Depth [m]')
        # Mark current depth
        current_flow = channel.flow_rate()
        ax2.scatter([current_flow], [channel.depth], color='red', s=100, zorder=5)
        ax2.annotate(f'({current_flow:.2f}, {channel.depth:.2f})',
                     (current_flow, channel.depth),
                     textcoords="offset points", xytext=(10, 10), color='red')

    ax2.set_title('Flow rate curve')
    ax2.set_xlim(left=0)

    # Synchronize Y axes between cross-section and flow rate curve
    y_min = min(min(elevations), min(depths))
    y_max = max(max(elevations), max(depths))
    margin = (y_max - y_min) * 0.05
    ax1.set_ylim(y_min - margin, y_max + margin)
    ax2.set_ylim(y_min - margin, y_max + margin)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    fig.tight_layout()
