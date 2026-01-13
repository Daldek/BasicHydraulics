"""
Culvert hydraulic calculations module.

Implements FHWA HY-8/HDS-5 methodology for culvert flow analysis
with automatic inlet/outlet control regime selection.

Supports optional downstream channel integration for iterative
tailwater calculation based on channel rating curve.

Classes:
    Culvert: Abstract base class for culvert calculations
    CircularCulvert: Circular (pipe) culvert
    BoxCulvert: Rectangular (box) culvert
    PipeArchCulvert: Pipe-arch (elliptical) culvert

Functions:
    plot_culvert: Visualize culvert cross-section and rating curve

Example:
    Basic usage with fixed tailwater::

        culvert = CircularCulvert(
            diameter=1.2, length=30, slope=0.01,
            headwater=2.0, tailwater=0.5
        )
        Q, regime = culvert.controlling_flow()

    With downstream channel for iterative tailwater::

        from basichydraulics.channel import RectangularChannel

        channel = RectangularChannel(
            depth=2.0, width=5.0, inclination=0.001, roughness_n=0.03
        )
        culvert = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=2.0,
            downstream_channel=channel
        )
        Q, regime = culvert.controlling_flow()
        # culvert.tailwater now contains calculated value
"""

import math
from abc import ABC, abstractmethod

from .structure import g  # Standard gravity 9.80665 m/s²


class Culvert(ABC):
    """
    Abstract base class for culvert hydraulic calculations.

    Implements FHWA HY-8/HDS-5 methodology for inlet and outlet control.

    Supports optional downstream channel integration: when a Channel object
    is provided via `downstream_channel`, tailwater is calculated iteratively
    from the channel's rating curve instead of using a fixed value.

    Attributes:
        length (float): Culvert length in meters.
        slope (float): Culvert slope (dimensionless, m/m).
        inlet_type (str): Inlet geometry type.
        roughness_n (float): Manning's roughness coefficient.
        headwater (float): Headwater depth above inlet invert (m).
        tailwater (float): Tailwater depth above outlet invert (m).
            When downstream_channel is set, this value is updated
            by outlet_control_flow() to reflect calculated tailwater.
        Ke (float): Entrance loss coefficient.
        fluid_density (float): Fluid density in kg/m³ (default 1000).
        downstream_channel: Optional Channel object for iterative tailwater.
            If provided, tailwater is calculated from channel's rating curve.
        outlet_invert_offset (float): Elevation difference between culvert
            outlet invert and downstream channel bottom (m). Positive if
            outlet is above channel bottom, negative if below.
    """

    def __init__(
        self,
        length,
        slope,
        inlet_type='projecting',
        roughness_n=0.024,
        headwater=None,
        tailwater=0.0,
        Ke=None,
        fluid_density=1000,
        downstream_channel=None,
        outlet_invert_offset=0.0
    ):
        """
        Initialize a Culvert instance.

        Args:
            length (float): Culvert length in meters.
            slope (float): Culvert slope (m/m).
            inlet_type (str): Inlet geometry type.
            roughness_n (float): Manning's roughness coefficient.
            headwater (float): Headwater depth above inlet invert (m).
            tailwater (float): Tailwater depth above outlet invert (m).
            Ke (float): Entrance loss coefficient. If None, selected based on inlet_type.
            fluid_density (float): Fluid density in kg/m³.
            downstream_channel: Optional Channel object for iterative tailwater calculation.
            outlet_invert_offset (float): Elevation difference between culvert outlet
                invert and downstream channel bottom (m). Positive if outlet is above
                channel bottom, negative if below.
        """
        self.length = length
        self.slope = slope
        self.inlet_type = inlet_type
        self.roughness_n = roughness_n
        self.headwater = headwater
        self.tailwater = tailwater
        self.Ke = Ke
        self.fluid_density = fluid_density
        self.downstream_channel = downstream_channel
        self.outlet_invert_offset = outlet_invert_offset

        # Cache for computed values
        self._cross_sectional_area = None
        self._wetted_perimeter_full = None
        self._hydraulic_radius_full = None
        self._tailwater_curve = None
        self._tailwater_exceeded_capacity = False
        self._downstream_channel_bottom = None

    def _reset_cache(self):
        """Reset all cached computed values."""
        self._cross_sectional_area = None
        self._wetted_perimeter_full = None
        self._hydraulic_radius_full = None
        self._tailwater_exceeded_capacity = False

    def _build_tailwater_curve(self):
        """
        Build Q -> water depth lookup from downstream channel rating curve.

        For IrregularChannel, converts absolute water levels to depths above
        channel bottom. For regular channels, uses depth directly.

        Returns:
            list: Sorted list of (Q, depth_above_channel_bottom) tuples,
                  or None if no downstream channel.
        """
        if self._tailwater_curve is not None:
            return self._tailwater_curve

        if self.downstream_channel is None:
            return None

        rating = self.downstream_channel.calculate_flow_rates()

        # Check if this is IrregularChannel (has water_level and elevations)
        if hasattr(self.downstream_channel, 'elevations'):
            # IrregularChannel - keys are absolute water levels
            channel_bottom = min(self.downstream_channel.elevations)
            self._downstream_channel_bottom = channel_bottom
            # Convert water_level to depth above channel bottom
            curve = [(q, wl - channel_bottom) for wl, q in rating.items() if q > 0]
        else:
            # Regular channel - keys are depths from 0
            self._downstream_channel_bottom = 0.0
            curve = [(q, h) for h, q in rating.items() if q > 0]

        curve.sort(key=lambda x: x[0])

        self._tailwater_curve = curve
        return curve

    def _interpolate_tailwater(self, Q):
        """
        Interpolate tailwater depth for given discharge from downstream channel.

        Calculates water depth above culvert outlet invert by:
        1. Getting water depth above channel bottom from rating curve
        2. Subtracting outlet_invert_offset to account for elevation difference

        Args:
            Q (float): Discharge in m³/s.

        Returns:
            float: Tailwater depth above culvert outlet invert (m).
                   Returns original tailwater if no downstream channel is defined.
                   Returns 0 if water level is below culvert outlet.
        """
        curve = self._build_tailwater_curve()
        if curve is None:
            return self.tailwater

        if Q <= 0:
            return max(0.0, -self.outlet_invert_offset)

        qs = [item[0] for item in curve]
        hs = [item[1] for item in curve]

        # Interpolate depth above channel bottom
        if Q <= qs[0]:
            depth_above_channel = hs[0]
        elif Q >= qs[-1]:
            self._tailwater_exceeded_capacity = True
            depth_above_channel = hs[-1]
        else:
            # Linear interpolation
            depth_above_channel = hs[-1]
            for i in range(len(qs) - 1):
                if qs[i] <= Q <= qs[i + 1]:
                    t = (Q - qs[i]) / (qs[i + 1] - qs[i])
                    depth_above_channel = hs[i] + t * (hs[i + 1] - hs[i])
                    break

        # Convert to depth above culvert outlet invert
        # tailwater = depth_above_channel - offset
        # If offset > 0 (outlet above channel bottom): tailwater < channel depth
        # If offset < 0 (outlet below channel bottom): tailwater > channel depth
        tailwater = depth_above_channel - self.outlet_invert_offset

        # Tailwater cannot be negative (water below outlet invert)
        return max(0.0, tailwater)

    def _outlet_control_flow_for_tailwater(self, tw):
        """
        Calculate outlet control flow for a specific tailwater value.

        Args:
            tw (float): Tailwater depth in meters.

        Returns:
            float: Flow rate in m³/s under outlet control.
        """
        if self.headwater is None or self.headwater <= 0:
            return 0.0

        A = self.cross_sectional_area()
        R = self.hydraulic_radius_full()

        # Total available head (datum at outlet invert)
        H_total = self.headwater + self.slope * self.length - tw

        if H_total <= 0:
            return 0.0

        # Loss coefficients
        Ke = self.Ke if self.Ke is not None else 0.5
        Kf = (2 * g * self.roughness_n**2 * self.length) / math.pow(R, 4.0 / 3.0)
        Ko = 1.0
        K_total = Ke + Kf + Ko

        Q = A * math.sqrt(2 * g * H_total / K_total)
        return Q

    def tailwater_exceeded_capacity(self):
        """
        Check if discharge exceeded downstream channel capacity.

        Returns:
            bool: True if Q exceeded max capacity of downstream channel.
        """
        return self._tailwater_exceeded_capacity

    @abstractmethod
    def cross_sectional_area(self):
        """
        Calculate full cross-sectional area of the culvert barrel.

        Returns:
            float: Cross-sectional area in m².
        """
        pass

    @abstractmethod
    def wetted_perimeter_full(self):
        """
        Calculate wetted perimeter when flowing full.

        Returns:
            float: Wetted perimeter in m.
        """
        pass

    @abstractmethod
    def rise(self):
        """
        Vertical height of the culvert opening.

        Returns:
            float: Rise (height) in meters.
        """
        pass

    @abstractmethod
    def span(self):
        """
        Horizontal width of the culvert opening.

        Returns:
            float: Span (width) in meters.
        """
        pass

    @abstractmethod
    def get_inlet_coefficients(self):
        """
        Return inlet control coefficients for this geometry/inlet type.

        Returns:
            dict: Dictionary with keys 'K', 'M', 'c', 'c_sub', 'Y', 'Ks', 'transition_hw'.
        """
        pass

    @abstractmethod
    def get_culvert_profile(self):
        """
        Return dictionary describing culvert geometry and parameters.

        Returns:
            dict: Profile information including type, dimensions, and parameters.
        """
        pass

    @abstractmethod
    def flow_area_at_depth(self, depth):
        """
        Calculate flow area at a given water depth.

        For partial flow (depth < rise), returns the wetted cross-sectional
        area. For full flow (depth >= rise), returns full cross-sectional area.

        Args:
            depth (float): Water depth from invert (m).

        Returns:
            float: Flow area in m².
        """
        pass

    def hydraulic_radius_full(self):
        """
        Calculate hydraulic radius when flowing full.

        Returns:
            float: Hydraulic radius in meters (A/P).
        """
        if self._hydraulic_radius_full is None:
            self._hydraulic_radius_full = (
                self.cross_sectional_area() / self.wetted_perimeter_full()
            )
        return self._hydraulic_radius_full

    def inlet_control_flow(self):
        """
        Calculate flow rate under inlet control conditions.

        Uses orifice flow equation with discharge coefficient (Cd) that
        varies based on inlet geometry. For partial flow (HW < D), uses
        the actual wetted flow area instead of full cross-section.

        The effective head transitions smoothly from headwater depth
        (unsubmerged) to head above centroid (submerged).

        Returns:
            float: Flow rate in m³/s under inlet control.
        """
        if self.headwater is None or self.headwater <= 0:
            return 0.0

        D = self.rise()
        A_full = self.cross_sectional_area()
        Cd = self.get_inlet_coefficients()['Cd']

        # Determine flow area based on submergence
        if self.headwater < D:
            # Partial flow - use actual wetted area at water depth
            A_flow = self.flow_area_at_depth(self.headwater)
        else:
            # Full flow - use full cross-sectional area
            A_flow = A_full

        # Effective head for orifice equation
        # Smoothly transitions from HW (unsubmerged) to HW - D/2 (submerged)
        # Transition region: 0.8D < HW < 1.5D

        if self.headwater <= 0.8 * D:
            # Unsubmerged - weir-like, head at water surface
            centroid_depth = 0
        elif self.headwater >= 1.5 * D:
            # Fully submerged - orifice, head above centroid
            centroid_depth = D / 2
        else:
            # Transition zone - linear interpolation
            fraction = (self.headwater - 0.8 * D) / (0.7 * D)
            centroid_depth = fraction * D / 2

        effective_head = self.headwater - centroid_depth

        if effective_head <= 0:
            return 0.0

        Q = Cd * A_flow * math.sqrt(2 * g * effective_head)
        return Q

    def outlet_control_flow(self, max_iter=50, tolerance=0.001):
        """
        Calculate flow rate under outlet control conditions.

        Uses energy equation balancing headwater against barrel friction
        plus entrance and exit losses.

        If a downstream_channel is defined, iteratively calculates tailwater
        from the channel's rating curve until convergence.

        Args:
            max_iter (int): Maximum iterations for tailwater convergence.
            tolerance (float): Convergence tolerance for Q (m³/s).

        Returns:
            float: Flow rate in m³/s under outlet control.
        """
        if self.headwater is None or self.headwater <= 0:
            return 0.0

        # If no downstream channel, use fixed tailwater
        if self.downstream_channel is None:
            return self._outlet_control_flow_for_tailwater(self.tailwater)

        # Iterative calculation with downstream channel
        # Start with current tailwater or 0
        tw = self.tailwater if self.tailwater else 0.0
        Q = self._outlet_control_flow_for_tailwater(tw)

        for _ in range(max_iter):
            # Get tailwater from downstream channel for current Q
            tw_new = self._interpolate_tailwater(Q)

            # Calculate new Q with updated tailwater
            Q_new = self._outlet_control_flow_for_tailwater(tw_new)

            # Check convergence
            if abs(Q_new - Q) < tolerance:
                self.tailwater = tw_new
                return Q_new

            Q = Q_new
            tw = tw_new

        # No convergence - use last values
        self.tailwater = tw
        return Q

    def controlling_flow(self):
        """
        Determine controlling flow regime and return flow rate.

        The controlling regime is the one that produces LOWER discharge
        for a given headwater (i.e., the bottleneck).

        Returns:
            tuple: (flow_rate_m3s, regime_str) where regime is 'inlet' or 'outlet'.
        """
        Q_inlet = self.inlet_control_flow()
        Q_outlet = self.outlet_control_flow()

        if Q_inlet <= Q_outlet:
            return (Q_inlet, 'inlet')
        else:
            return (Q_outlet, 'outlet')

    def velocity_inlet(self):
        """
        Calculate velocity at culvert inlet.

        Returns:
            float: Inlet velocity in m/s.
        """
        Q, _ = self.controlling_flow()
        A = self.cross_sectional_area()
        if A > 0 and Q > 0:
            return Q / A
        return 0.0

    def velocity_outlet(self):
        """
        Calculate velocity at culvert outlet.

        For full flow, outlet velocity equals inlet velocity.

        Returns:
            float: Outlet velocity in m/s.
        """
        return self.velocity_inlet()

    def calculate_rating_curve(self, hw_min=None, hw_max=None, step=0.1):
        """
        Generate headwater-discharge rating curve.

        Args:
            hw_min (float): Minimum headwater depth (m). Defaults to step.
            hw_max (float): Maximum headwater depth (m). Defaults to 3 * rise.
            step (float): Increment step (m).

        Returns:
            dict: Dictionary with keys 'headwater', 'discharge', 'regime',
                  'velocity_inlet', 'velocity_outlet'.
        """
        if hw_min is None:
            hw_min = step
        if hw_max is None:
            hw_max = 3 * self.rise()

        original_hw = self.headwater
        results = {
            'headwater': [],
            'discharge': [],
            'regime': [],
            'velocity_inlet': [],
            'velocity_outlet': []
        }

        hw = hw_min
        while hw <= hw_max + step / 2:
            self.headwater = round(hw, 3)
            self._reset_cache()

            if hw <= 0:
                hw += step
                continue

            Q, regime = self.controlling_flow()
            results['headwater'].append(round(hw, 3))
            results['discharge'].append(round(Q, 4))
            results['regime'].append(regime)
            results['velocity_inlet'].append(round(self.velocity_inlet(), 3))
            results['velocity_outlet'].append(round(self.velocity_outlet(), 3))

            hw += step

        self.headwater = original_hw
        self._reset_cache()
        return results


class CircularCulvert(Culvert):
    """
    Circular (pipe) culvert.

    Attributes:
        diameter (float): Internal pipe diameter in meters.
    """

    # Discharge coefficients (Cd) for inlet control - dimensionless
    # Based on orifice flow theory and empirical data
    # Cd accounts for contraction and energy losses at inlet
    INLET_COEFFICIENTS = {
        'projecting': {'Cd': 0.50},      # Pipe projecting from fill
        'headwall': {'Cd': 0.62},        # Square edge in headwall
        'headwall_beveled': {'Cd': 0.70}, # Beveled edge in headwall
        'mitered': {'Cd': 0.53},         # Mitered to slope
        'wingwall_30': {'Cd': 0.65},     # Wingwalls at 30°
        'wingwall_45': {'Cd': 0.68},     # Wingwalls at 45°
    }

    # Entrance loss coefficients (Ke) for outlet control
    ENTRANCE_LOSS = {
        'projecting': 0.78,
        'headwall': 0.50,
        'headwall_beveled': 0.25,
        'mitered': 0.70,
        'wingwall_30': 0.40,
        'wingwall_45': 0.35
    }

    def __init__(
        self,
        diameter,
        length,
        slope,
        inlet_type='projecting',
        roughness_n=0.024,
        headwater=None,
        tailwater=0.0,
        Ke=None,
        fluid_density=1000,
        downstream_channel=None,
        outlet_invert_offset=0.0
    ):
        """
        Initialize a CircularCulvert instance.

        Args:
            diameter (float): Internal pipe diameter in meters.
            length (float): Culvert length in meters.
            slope (float): Culvert slope (m/m).
            inlet_type (str): Inlet geometry type. Options: 'projecting',
                'headwall', 'headwall_beveled', 'mitered'.
            roughness_n (float): Manning's roughness coefficient.
                Default 0.024 for corrugated metal pipe.
            headwater (float): Headwater depth above inlet invert (m).
            tailwater (float): Tailwater depth above outlet invert (m).
            Ke (float): Entrance loss coefficient. If None, auto-selected.
            fluid_density (float): Fluid density in kg/m³.
            downstream_channel: Optional Channel object for iterative tailwater.
            outlet_invert_offset (float): Elevation of outlet invert above
                downstream channel bottom (m). Positive if outlet is higher.
        """
        self.diameter = diameter

        # Auto-select Ke if not provided
        if Ke is None:
            Ke = self.ENTRANCE_LOSS.get(inlet_type, 0.50)

        super().__init__(
            length=length,
            slope=slope,
            inlet_type=inlet_type,
            roughness_n=roughness_n,
            headwater=headwater,
            tailwater=tailwater,
            Ke=Ke,
            fluid_density=fluid_density,
            downstream_channel=downstream_channel,
            outlet_invert_offset=outlet_invert_offset
        )

    def cross_sectional_area(self):
        """Calculate circular cross-sectional area."""
        if self._cross_sectional_area is None:
            self._cross_sectional_area = math.pi * self.diameter**2 / 4
        return self._cross_sectional_area

    def wetted_perimeter_full(self):
        """Calculate circular wetted perimeter (circumference)."""
        if self._wetted_perimeter_full is None:
            self._wetted_perimeter_full = math.pi * self.diameter
        return self._wetted_perimeter_full

    def rise(self):
        """Return culvert rise (diameter for circular)."""
        return self.diameter

    def span(self):
        """Return culvert span (diameter for circular)."""
        return self.diameter

    def get_inlet_coefficients(self):
        """Return inlet control coefficients for circular culvert."""
        return self.INLET_COEFFICIENTS.get(
            self.inlet_type,
            self.INLET_COEFFICIENTS['projecting']
        )

    def get_culvert_profile(self):
        """Return circular culvert profile information."""
        return {
            'type': 'circular',
            'diameter': self.diameter,
            'length': self.length,
            'slope': self.slope,
            'inlet_type': self.inlet_type,
            'roughness_n': self.roughness_n,
            'Ke': self.Ke,
            'area': self.cross_sectional_area()
        }

    def flow_area_at_depth(self, depth):
        """
        Calculate flow area for partial depth in circular culvert.

        Uses circular segment formula:
        θ = 2 × arccos(1 - 2h/D)
        A = (D²/8) × (θ - sin(θ))

        Args:
            depth (float): Water depth from invert (m).

        Returns:
            float: Flow area in m².
        """
        if depth <= 0:
            return 0.0
        if depth >= self.diameter:
            return self.cross_sectional_area()

        # Circular segment area
        D = self.diameter
        theta = 2 * math.acos(1 - 2 * depth / D)
        area = (D**2 / 8) * (theta - math.sin(theta))
        return area


class BoxCulvert(Culvert):
    """
    Rectangular (box) culvert.

    Attributes:
        width (float): Internal width (span) in meters.
        height (float): Internal height (rise) in meters.
    """

    # Discharge coefficients (Cd) for inlet control - dimensionless
    INLET_COEFFICIENTS = {
        'headwall': {'Cd': 0.62},         # Square edge in headwall
        'headwall_beveled': {'Cd': 0.70}, # Beveled edge
        'wingwall_30': {'Cd': 0.65},      # Wingwalls at 30°
        'wingwall_45': {'Cd': 0.68},      # Wingwalls at 45°
        'wingwall_90': {'Cd': 0.62},      # Parallel wingwalls (90°)
    }

    # Entrance loss coefficients (Ke) for outlet control
    ENTRANCE_LOSS = {
        'headwall': 0.50,
        'headwall_beveled': 0.20,
        'wingwall_30': 0.40,
        'wingwall_45': 0.35,
        'wingwall_90': 0.50
    }

    def __init__(
        self,
        width,
        height,
        length,
        slope,
        inlet_type='headwall',
        roughness_n=0.012,
        headwater=None,
        tailwater=0.0,
        Ke=None,
        fluid_density=1000,
        downstream_channel=None,
        outlet_invert_offset=0.0
    ):
        """
        Initialize a BoxCulvert instance.

        Args:
            width (float): Internal width (span) in meters.
            height (float): Internal height (rise) in meters.
            length (float): Culvert length in meters.
            slope (float): Culvert slope (m/m).
            inlet_type (str): Inlet geometry type. Options: 'headwall',
                'headwall_beveled', 'wingwall_30', 'wingwall_45', 'wingwall_90'.
            roughness_n (float): Manning's roughness coefficient.
                Default 0.012 for concrete.
            headwater (float): Headwater depth above inlet invert (m).
            tailwater (float): Tailwater depth above outlet invert (m).
            Ke (float): Entrance loss coefficient. If None, auto-selected.
            fluid_density (float): Fluid density in kg/m³.
            downstream_channel: Optional Channel object for iterative tailwater.
            outlet_invert_offset (float): Elevation of outlet invert above
                downstream channel bottom (m). Positive if outlet is higher.
        """
        self.width = width
        self.height = height

        if Ke is None:
            Ke = self.ENTRANCE_LOSS.get(inlet_type, 0.50)

        super().__init__(
            length=length,
            slope=slope,
            inlet_type=inlet_type,
            roughness_n=roughness_n,
            headwater=headwater,
            tailwater=tailwater,
            Ke=Ke,
            fluid_density=fluid_density,
            downstream_channel=downstream_channel,
            outlet_invert_offset=outlet_invert_offset
        )

    def cross_sectional_area(self):
        """Calculate rectangular cross-sectional area."""
        if self._cross_sectional_area is None:
            self._cross_sectional_area = self.width * self.height
        return self._cross_sectional_area

    def wetted_perimeter_full(self):
        """Calculate rectangular wetted perimeter."""
        if self._wetted_perimeter_full is None:
            self._wetted_perimeter_full = 2 * (self.width + self.height)
        return self._wetted_perimeter_full

    def rise(self):
        """Return culvert rise (height)."""
        return self.height

    def span(self):
        """Return culvert span (width)."""
        return self.width

    def get_inlet_coefficients(self):
        """Return inlet control coefficients for box culvert."""
        return self.INLET_COEFFICIENTS.get(
            self.inlet_type,
            self.INLET_COEFFICIENTS['headwall']
        )

    def get_culvert_profile(self):
        """Return box culvert profile information."""
        return {
            'type': 'box',
            'width': self.width,
            'height': self.height,
            'length': self.length,
            'slope': self.slope,
            'inlet_type': self.inlet_type,
            'roughness_n': self.roughness_n,
            'Ke': self.Ke,
            'area': self.cross_sectional_area()
        }

    def flow_area_at_depth(self, depth):
        """
        Calculate flow area for partial depth in box culvert.

        Simple rectangular area: A = width × depth

        Args:
            depth (float): Water depth from invert (m).

        Returns:
            float: Flow area in m².
        """
        if depth <= 0:
            return 0.0
        if depth >= self.height:
            return self.cross_sectional_area()

        return self.width * depth


class PipeArchCulvert(Culvert):
    """
    Pipe-arch (elliptical approximation) culvert.

    Pipe-arch has a flat bottom and arched top.
    Geometry is approximated as an ellipse for hydraulic calculations.

    Attributes:
        span_width (float): Horizontal width (span) in meters.
        rise_height (float): Vertical height (rise) in meters.
    """

    # Discharge coefficients (Cd) for inlet control - dimensionless
    # Slightly lower than circular due to less favorable hydraulic shape
    INLET_COEFFICIENTS = {
        'projecting': {'Cd': 0.48},  # Projecting from fill
        'headwall': {'Cd': 0.60},    # Square edge in headwall
        'mitered': {'Cd': 0.52},     # Mitered to slope
    }

    # Entrance loss coefficients (Ke) for outlet control
    ENTRANCE_LOSS = {
        'projecting': 0.78,
        'headwall': 0.50,
        'mitered': 0.70
    }

    def __init__(
        self,
        span_width,
        rise_height,
        length,
        slope,
        inlet_type='projecting',
        roughness_n=0.024,
        headwater=None,
        tailwater=0.0,
        Ke=None,
        fluid_density=1000,
        downstream_channel=None,
        outlet_invert_offset=0.0
    ):
        """
        Initialize a PipeArchCulvert instance.

        Args:
            span_width (float): Horizontal width (span) in meters.
            rise_height (float): Vertical height (rise) in meters.
            length (float): Culvert length in meters.
            slope (float): Culvert slope (m/m).
            inlet_type (str): Inlet geometry type. Options: 'projecting',
                'headwall', 'mitered'.
            roughness_n (float): Manning's roughness coefficient.
                Default 0.024 for corrugated metal.
            headwater (float): Headwater depth above inlet invert (m).
            tailwater (float): Tailwater depth above outlet invert (m).
            Ke (float): Entrance loss coefficient. If None, auto-selected.
            fluid_density (float): Fluid density in kg/m³.
            downstream_channel: Optional Channel object for iterative tailwater.
            outlet_invert_offset (float): Elevation of outlet invert above
                downstream channel bottom (m). Positive if outlet is higher.
        """
        self._span = span_width
        self._rise = rise_height

        if Ke is None:
            Ke = self.ENTRANCE_LOSS.get(inlet_type, 0.50)

        super().__init__(
            length=length,
            slope=slope,
            inlet_type=inlet_type,
            roughness_n=roughness_n,
            headwater=headwater,
            tailwater=tailwater,
            Ke=Ke,
            fluid_density=fluid_density,
            downstream_channel=downstream_channel,
            outlet_invert_offset=outlet_invert_offset
        )

    def cross_sectional_area(self):
        """Calculate elliptical cross-sectional area approximation."""
        if self._cross_sectional_area is None:
            # Approximate as ellipse: A = pi * a * b
            # where a = span/2, b = rise/2
            self._cross_sectional_area = math.pi * self._span * self._rise / 4
        return self._cross_sectional_area

    def wetted_perimeter_full(self):
        """Calculate elliptical wetted perimeter using Ramanujan approximation."""
        if self._wetted_perimeter_full is None:
            a = self._span / 2
            b = self._rise / 2
            # Ramanujan approximation for ellipse perimeter
            h = ((a - b)**2) / ((a + b)**2)
            self._wetted_perimeter_full = (
                math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
            )
        return self._wetted_perimeter_full

    def rise(self):
        """Return culvert rise (height)."""
        return self._rise

    def span(self):
        """Return culvert span (width)."""
        return self._span

    def get_inlet_coefficients(self):
        """Return inlet control coefficients for pipe-arch culvert."""
        return self.INLET_COEFFICIENTS.get(
            self.inlet_type,
            self.INLET_COEFFICIENTS['projecting']
        )

    def get_culvert_profile(self):
        """Return pipe-arch culvert profile information."""
        return {
            'type': 'pipe_arch',
            'span': self._span,
            'rise': self._rise,
            'length': self.length,
            'slope': self.slope,
            'inlet_type': self.inlet_type,
            'roughness_n': self.roughness_n,
            'Ke': self.Ke,
            'area': self.cross_sectional_area()
        }

    def flow_area_at_depth(self, depth):
        """
        Calculate flow area for partial depth in pipe-arch culvert.

        Uses ellipse segment approximation. For depth h in ellipse with
        semi-axes a (horizontal) and b (vertical):
        A = a*b * arccos(1 - h/b) - a*(b-h)*sqrt(2*h/b - (h/b)^2)

        Args:
            depth (float): Water depth from invert (m).

        Returns:
            float: Flow area in m².
        """
        if depth <= 0:
            return 0.0
        if depth >= self._rise:
            return self.cross_sectional_area()

        # Ellipse semi-axes
        a = self._span / 2  # horizontal
        b = self._rise / 2  # vertical

        # Ellipse segment area (from bottom)
        # Transform: y measured from bottom, so y_center = b
        h = depth
        h_ratio = h / (2 * b)  # h / rise = h / (2b)

        if h_ratio >= 1.0:
            return self.cross_sectional_area()

        # Area of ellipse segment from y=0 to y=h
        # A = a*b*arccos(1-h/b) - a*(b-h)*sqrt(h*(2b-h))/b
        # Simplified for y from bottom of ellipse:
        y_norm = h / self._rise  # normalized depth 0 to 1
        theta = math.acos(1 - 2 * y_norm)
        area = (a * b) * (theta - math.sin(theta) * math.cos(theta))
        return area


def plot_culvert(culvert, hw_max=None, title=None):
    """
    Visualize culvert rating curve and cross-section.

    Creates a figure with:
    - Left panel: Cross-section view with headwater level
    - Right panel: Rating curve (Q vs HW) as continuous line
      with current operating point marked

    Args:
        culvert: Culvert object (any subclass).
        hw_max (float): Maximum headwater for rating curve.
        title (str): Optional plot title.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Circle, Rectangle, Ellipse

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    profile = culvert.get_culvert_profile()

    # Left panel: Cross-section
    if profile['type'] == 'circular':
        circle = Circle(
            (0, profile['diameter'] / 2),
            profile['diameter'] / 2,
            fill=False, edgecolor='black', linewidth=2
        )
        ax1.add_patch(circle)
        ax1.set_xlim(-profile['diameter'], profile['diameter'])
        ax1.set_ylim(-0.2, profile['diameter'] * 1.5)

    elif profile['type'] == 'box':
        rect = Rectangle(
            (-profile['width'] / 2, 0),
            profile['width'], profile['height'],
            fill=False, edgecolor='black', linewidth=2
        )
        ax1.add_patch(rect)
        ax1.set_xlim(-profile['width'], profile['width'])
        ax1.set_ylim(-0.2, profile['height'] * 1.5)

    elif profile['type'] == 'pipe_arch':
        ellipse = Ellipse(
            (0, profile['rise'] / 2),
            profile['span'], profile['rise'],
            fill=False, edgecolor='black', linewidth=2
        )
        ax1.add_patch(ellipse)
        ax1.set_xlim(-profile['span'], profile['span'])
        ax1.set_ylim(-0.2, profile['rise'] * 1.5)

    # Draw headwater level
    if culvert.headwater and culvert.headwater > 0:
        hw_line = ax1.axhline(
            y=culvert.headwater, color='blue',
            linestyle='--', linewidth=1.5,
            label=f'HW = {culvert.headwater:.2f} m'
        )
        # Fill water area
        xlim = ax1.get_xlim()
        ax1.fill_between(
            [xlim[0], xlim[1]], [0, 0],
            [min(culvert.headwater, culvert.rise())] * 2,
            alpha=0.3, color='blue'
        )
        ax1.legend()

    ax1.set_xlabel('Width [m]')
    ax1.set_ylabel('Height [m]')
    ax1.set_title('Cross-section')
    ax1.set_aspect('equal')
    ax1.grid(True)

    # Right panel: Rating curve
    if hw_max is None:
        hw_max = max(
            culvert.headwater * 2 if culvert.headwater else 2,
            culvert.rise() * 2
        )

    rating = culvert.calculate_rating_curve(hw_max=hw_max, step=0.05)

    # Plot rating curve as single continuous line
    ax2.plot(rating['discharge'], rating['headwater'], 'b-', linewidth=2,
             label='Rating curve')

    # Mark current operating point
    if culvert.headwater and culvert.headwater > 0:
        Q, regime = culvert.controlling_flow()
        ax2.scatter([Q], [culvert.headwater], color='green',
                    s=150, zorder=5, marker='*')
        ax2.annotate(
            f'Q={Q:.3f} m³/s\n({regime})',
            (Q, culvert.headwater),
            textcoords="offset points", xytext=(10, 10)
        )

    ax2.set_xlabel('Discharge [m³/s]')
    ax2.set_ylabel('Headwater [m]')
    ax2.set_title('Rating Curve')
    ax2.legend(loc='lower right')
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.grid(True)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    fig.tight_layout()
    return fig
