"""
Unit tests for culvert hydraulic calculations module.

Tests cover:
- Geometry calculations (area, perimeter) for all culvert types
- Inlet control flow calculations (unsubmerged and submerged)
- Outlet control flow calculations
- Controlling flow regime selection
- Rating curve generation
- Edge cases and validation
"""

import unittest
import math

from basichydraulics.culvert import (
    CircularCulvert,
    BoxCulvert,
    PipeArchCulvert,
    plot_culvert,
    g
)
from basichydraulics.channel import RectangularChannel, TrapezoidalChannel, IrregularChannel


class TestCircularCulvert(unittest.TestCase):
    """Tests for circular (pipe) culvert calculations."""

    def test_cross_sectional_area(self):
        """Test circular area calculation: A = pi * D^2 / 4"""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        expected = math.pi * 1.2**2 / 4
        self.assertAlmostEqual(culvert.cross_sectional_area(), expected, places=4)

    def test_wetted_perimeter(self):
        """Test circular perimeter calculation: P = pi * D"""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        expected = math.pi * 1.2
        self.assertAlmostEqual(culvert.wetted_perimeter_full(), expected, places=4)

    def test_hydraulic_radius(self):
        """Test hydraulic radius: R = A / P = D / 4"""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        expected = 1.2 / 4  # For circular pipe, R = D/4
        self.assertAlmostEqual(culvert.hydraulic_radius_full(), expected, places=4)

    def test_rise_and_span(self):
        """Test rise and span equal diameter for circular."""
        culvert = CircularCulvert(
            diameter=1.5,
            length=30,
            slope=0.01,
            headwater=2.0
        )
        self.assertEqual(culvert.rise(), 1.5)
        self.assertEqual(culvert.span(), 1.5)

    def test_inlet_control_flow_unsubmerged(self):
        """Test inlet control with HW/D < 1.2 (unsubmerged/transitional)."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            inlet_type='headwall',
            headwater=0.8  # HW/D = 0.67 < 1.2
        )
        Q = culvert.inlet_control_flow()
        self.assertGreater(Q, 0)
        self.assertLess(Q, 10)  # Sanity check for reasonable flow

    def test_inlet_control_flow_submerged(self):
        """Test inlet control with HW/D >= 1.2 (submerged)."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            inlet_type='headwall',
            headwater=2.0  # HW/D = 1.67 >= 1.2
        )
        Q = culvert.inlet_control_flow()
        self.assertGreater(Q, 0)
        self.assertLess(Q, 10)  # Sanity check for reasonable flow

    def test_outlet_control_flow(self):
        """Test outlet control calculation produces positive flow."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=2.0,
            tailwater=0.5
        )
        Q = culvert.outlet_control_flow()
        self.assertGreater(Q, 0)

    def test_controlling_flow_returns_minimum(self):
        """Controlling flow should be the minimum of inlet/outlet."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=2.0,
            tailwater=0.5
        )
        Q, regime = culvert.controlling_flow()
        Q_inlet = culvert.inlet_control_flow()
        Q_outlet = culvert.outlet_control_flow()

        self.assertEqual(Q, min(Q_inlet, Q_outlet))
        self.assertIn(regime, ['inlet', 'outlet'])

    def test_rating_curve_monotonic(self):
        """Rating curve discharge should increase with headwater."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.0
        )
        rating = culvert.calculate_rating_curve(hw_min=0.1, hw_max=3.0, step=0.1)

        # Check monotonically increasing (or equal)
        for i in range(1, len(rating['discharge'])):
            self.assertGreaterEqual(rating['discharge'][i], rating['discharge'][i - 1])

    def test_entrance_loss_coefficient_auto_selection(self):
        """Test automatic Ke selection based on inlet type."""
        culvert_proj = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=1.0,
            inlet_type='projecting'
        )
        culvert_head = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=1.0,
            inlet_type='headwall'
        )
        culvert_bevel = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=1.0,
            inlet_type='headwall_beveled'
        )

        # Projecting has highest Ke, beveled has lowest
        self.assertGreater(culvert_proj.Ke, culvert_head.Ke)
        self.assertGreater(culvert_head.Ke, culvert_bevel.Ke)

    def test_velocity_calculation(self):
        """Test velocity is Q/A."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=2.0
        )
        Q, _ = culvert.controlling_flow()
        A = culvert.cross_sectional_area()
        expected_v = Q / A if A > 0 else 0

        self.assertAlmostEqual(culvert.velocity_inlet(), expected_v, places=3)

    def test_culvert_profile(self):
        """Test get_culvert_profile returns correct structure."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5,
            inlet_type='headwall'
        )
        profile = culvert.get_culvert_profile()

        self.assertEqual(profile['type'], 'circular')
        self.assertEqual(profile['diameter'], 1.2)
        self.assertEqual(profile['length'], 30)
        self.assertEqual(profile['slope'], 0.01)
        self.assertEqual(profile['inlet_type'], 'headwall')
        self.assertIn('area', profile)


class TestBoxCulvert(unittest.TestCase):
    """Tests for box (rectangular) culvert calculations."""

    def test_cross_sectional_area(self):
        """Test rectangular area calculation: A = W * H"""
        culvert = BoxCulvert(
            width=2.0,
            height=1.5,
            length=30,
            slope=0.01,
            headwater=2.0
        )
        self.assertEqual(culvert.cross_sectional_area(), 3.0)

    def test_wetted_perimeter(self):
        """Test rectangular perimeter: P = 2*(W + H)"""
        culvert = BoxCulvert(
            width=2.0,
            height=1.5,
            length=30,
            slope=0.01,
            headwater=2.0
        )
        self.assertEqual(culvert.wetted_perimeter_full(), 7.0)

    def test_hydraulic_radius(self):
        """Test hydraulic radius for box culvert."""
        culvert = BoxCulvert(
            width=2.0,
            height=1.5,
            length=30,
            slope=0.01,
            headwater=2.0
        )
        expected = 3.0 / 7.0  # A / P
        self.assertAlmostEqual(culvert.hydraulic_radius_full(), expected, places=4)

    def test_rise_and_span(self):
        """Test rise equals height, span equals width."""
        culvert = BoxCulvert(
            width=2.5,
            height=1.8,
            length=30,
            slope=0.01,
            headwater=2.0
        )
        self.assertEqual(culvert.rise(), 1.8)
        self.assertEqual(culvert.span(), 2.5)

    def test_inlet_types(self):
        """Test different inlet types produce different results."""
        base_params = {
            'width': 2.0,
            'height': 1.5,
            'length': 30,
            'slope': 0.01,
            'headwater': 2.0
        }

        culvert_head = BoxCulvert(**base_params, inlet_type='headwall')
        culvert_wing = BoxCulvert(**base_params, inlet_type='wingwall_45')

        Q_head, _ = culvert_head.controlling_flow()
        Q_wing, _ = culvert_wing.controlling_flow()

        # Different inlet types should give different flows
        self.assertNotEqual(Q_head, Q_wing)

    def test_culvert_profile(self):
        """Test get_culvert_profile returns correct structure."""
        culvert = BoxCulvert(
            width=2.0,
            height=1.5,
            length=30,
            slope=0.01,
            headwater=2.0
        )
        profile = culvert.get_culvert_profile()

        self.assertEqual(profile['type'], 'box')
        self.assertEqual(profile['width'], 2.0)
        self.assertEqual(profile['height'], 1.5)


class TestPipeArchCulvert(unittest.TestCase):
    """Tests for pipe-arch (elliptical) culvert calculations."""

    def test_cross_sectional_area(self):
        """Test elliptical area approximation: A = pi * span * rise / 4"""
        culvert = PipeArchCulvert(
            span_width=1.8,
            rise_height=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        expected = math.pi * 1.8 * 1.2 / 4
        self.assertAlmostEqual(culvert.cross_sectional_area(), expected, places=4)

    def test_wetted_perimeter_ramanujan(self):
        """Test Ramanujan approximation for ellipse perimeter."""
        culvert = PipeArchCulvert(
            span_width=2.0,
            rise_height=2.0,  # Circle case
            length=30,
            slope=0.01,
            headwater=1.5
        )
        # For a circle (span = rise), should be close to pi * D
        expected_circle = math.pi * 2.0
        self.assertAlmostEqual(culvert.wetted_perimeter_full(), expected_circle, places=2)

    def test_rise_and_span(self):
        """Test rise and span return correct values."""
        culvert = PipeArchCulvert(
            span_width=1.8,
            rise_height=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        self.assertEqual(culvert.rise(), 1.2)
        self.assertEqual(culvert.span(), 1.8)

    def test_culvert_profile(self):
        """Test get_culvert_profile returns correct structure."""
        culvert = PipeArchCulvert(
            span_width=1.8,
            rise_height=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        profile = culvert.get_culvert_profile()

        self.assertEqual(profile['type'], 'pipe_arch')
        self.assertEqual(profile['span'], 1.8)
        self.assertEqual(profile['rise'], 1.2)


class TestCulvertValidation(unittest.TestCase):
    """Tests for input validation and edge cases."""

    def test_zero_headwater_returns_zero_flow(self):
        """Zero headwater should produce zero flow."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=0.0
        )
        Q, _ = culvert.controlling_flow()
        self.assertEqual(Q, 0.0)

    def test_none_headwater_returns_zero_flow(self):
        """None headwater should produce zero flow."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=None
        )
        Q, _ = culvert.controlling_flow()
        self.assertEqual(Q, 0.0)

    def test_negative_headwater_returns_zero_flow(self):
        """Negative headwater should produce zero flow."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=-1.0
        )
        Q, _ = culvert.controlling_flow()
        self.assertEqual(Q, 0.0)

    def test_high_tailwater_reduces_flow(self):
        """High tailwater should reduce outlet control flow."""
        culvert_low_tw = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=2.0,
            tailwater=0.0
        )
        culvert_high_tw = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=2.0,
            tailwater=1.5
        )

        Q_low = culvert_low_tw.outlet_control_flow()
        Q_high = culvert_high_tw.outlet_control_flow()

        self.assertGreater(Q_low, Q_high)

    def test_adverse_slope(self):
        """Culvert should handle adverse (negative) slope."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=-0.01,  # Adverse slope
            headwater=2.0,
            tailwater=0.0
        )
        # Should still calculate (energy still available from headwater)
        Q, _ = culvert.controlling_flow()
        self.assertGreaterEqual(Q, 0)

    def test_steep_slope_increases_flow(self):
        """Steeper slope should increase outlet control capacity."""
        culvert_flat = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.001,
            headwater=2.0,
            tailwater=0.5
        )
        culvert_steep = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.05,
            headwater=2.0,
            tailwater=0.5
        )

        Q_flat = culvert_flat.outlet_control_flow()
        Q_steep = culvert_steep.outlet_control_flow()

        self.assertGreater(Q_steep, Q_flat)

    def test_custom_ke_override(self):
        """Custom Ke should override auto-selection."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=2.0,
            inlet_type='headwall',
            Ke=0.99  # Custom value
        )
        self.assertEqual(culvert.Ke, 0.99)


class TestRatingCurve(unittest.TestCase):
    """Tests for rating curve generation."""

    def test_rating_curve_keys(self):
        """Rating curve should contain all expected keys."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        rating = culvert.calculate_rating_curve(hw_max=3.0, step=0.5)

        self.assertIn('headwater', rating)
        self.assertIn('discharge', rating)
        self.assertIn('regime', rating)
        self.assertIn('velocity_inlet', rating)
        self.assertIn('velocity_outlet', rating)

    def test_rating_curve_lengths_match(self):
        """All rating curve lists should have same length."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        rating = culvert.calculate_rating_curve(hw_max=3.0, step=0.5)

        n = len(rating['headwater'])
        self.assertEqual(len(rating['discharge']), n)
        self.assertEqual(len(rating['regime']), n)
        self.assertEqual(len(rating['velocity_inlet']), n)
        self.assertEqual(len(rating['velocity_outlet']), n)

    def test_rating_curve_preserves_original_headwater(self):
        """Original headwater should be preserved after rating curve calculation."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        original_hw = culvert.headwater

        _ = culvert.calculate_rating_curve(hw_max=5.0, step=0.5)

        self.assertEqual(culvert.headwater, original_hw)

    def test_rating_curve_step_size(self):
        """Rating curve should respect step size."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        rating = culvert.calculate_rating_curve(hw_min=0.5, hw_max=1.5, step=0.5)

        # Should have points at 0.5, 1.0, 1.5 = 3 points
        self.assertEqual(len(rating['headwater']), 3)


class TestPlotCulvert(unittest.TestCase):
    """Tests for culvert visualization function."""

    def test_plot_returns_figure(self):
        """plot_culvert should return a matplotlib Figure."""
        import matplotlib.pyplot as plt

        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        fig = plot_culvert(culvert)

        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'savefig'))  # It's a Figure object
        plt.close(fig)

    def test_plot_with_title(self):
        """plot_culvert should accept custom title."""
        import matplotlib.pyplot as plt

        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5
        )
        fig = plot_culvert(culvert, title="Test Culvert")

        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_plot_all_culvert_types(self):
        """plot_culvert should work for all culvert types."""
        import matplotlib.pyplot as plt

        culverts = [
            CircularCulvert(diameter=1.2, length=30, slope=0.01, headwater=1.5),
            BoxCulvert(width=2.0, height=1.5, length=30, slope=0.01, headwater=2.0),
            PipeArchCulvert(span_width=1.8, rise_height=1.2, length=30, slope=0.01, headwater=1.5)
        ]

        for culvert in culverts:
            fig = plot_culvert(culvert)
            self.assertIsNotNone(fig)
            plt.close(fig)


class TestDownstreamChannel(unittest.TestCase):
    """Tests for downstream channel integration."""

    def test_downstream_channel_none_backward_compatible(self):
        """Without downstream_channel, behavior should be unchanged."""
        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=2.0,
            tailwater=0.5
        )
        self.assertIsNone(culvert.downstream_channel)
        Q = culvert.outlet_control_flow()
        self.assertGreater(Q, 0)

    def test_downstream_channel_rectangular(self):
        """Test iterative tailwater with rectangular channel."""
        # Create downstream channel
        channel = RectangularChannel(
            depth=2.0,
            width=5.0,
            inclination=0.001,
            roughness_n=0.03
        )

        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=2.0,
            tailwater=0.0,  # Will be calculated iteratively
            downstream_channel=channel
        )

        Q, regime = culvert.controlling_flow()
        self.assertGreater(Q, 0)
        # Tailwater should be updated based on channel
        self.assertGreater(culvert.tailwater, 0)

    def test_downstream_channel_trapezoidal(self):
        """Test iterative tailwater with trapezoidal channel."""
        channel = TrapezoidalChannel(
            depth=2.0,
            width=3.0,
            bank_inclination_m=1.5,
            inclination=0.001,
            roughness_n=0.03
        )

        culvert = BoxCulvert(
            width=2.0,
            height=1.5,
            length=30,
            slope=0.01,
            headwater=2.0,
            downstream_channel=channel
        )

        Q, _ = culvert.controlling_flow()
        self.assertGreater(Q, 0)
        self.assertGreater(culvert.tailwater, 0)

    def test_tailwater_converges(self):
        """Test that iterative calculation converges."""
        channel = RectangularChannel(
            depth=3.0,
            width=6.0,
            inclination=0.002,
            roughness_n=0.025
        )

        culvert = CircularCulvert(
            diameter=1.5,
            length=50,
            slope=0.005,
            headwater=3.0,
            downstream_channel=channel
        )

        # Run calculation multiple times - should get same result
        Q1, _ = culvert.controlling_flow()
        tw1 = culvert.tailwater

        culvert._tailwater_curve = None  # Reset cache
        Q2, _ = culvert.controlling_flow()
        tw2 = culvert.tailwater

        self.assertAlmostEqual(Q1, Q2, places=3)
        self.assertAlmostEqual(tw1, tw2, places=3)

    def test_downstream_channel_exceeds_capacity(self):
        """Test warning flag when Q exceeds channel capacity."""
        # Create a small channel that will be exceeded
        channel = RectangularChannel(
            depth=0.5,  # Small depth = low capacity
            width=1.0,
            inclination=0.001,
            roughness_n=0.03
        )

        culvert = CircularCulvert(
            diameter=2.0,  # Large culvert
            length=20,
            slope=0.02,
            headwater=4.0,  # High headwater = high Q
            downstream_channel=channel
        )

        Q, _ = culvert.controlling_flow()
        # Check if capacity exceeded flag is set
        self.assertTrue(culvert.tailwater_exceeded_capacity())

    def test_rating_curve_with_downstream_channel(self):
        """Test rating curve generation with downstream channel."""
        channel = RectangularChannel(
            depth=2.0,
            width=4.0,
            inclination=0.001,
            roughness_n=0.03
        )

        culvert = CircularCulvert(
            diameter=1.2,
            length=30,
            slope=0.01,
            headwater=1.5,
            downstream_channel=channel
        )

        rating = culvert.calculate_rating_curve(hw_min=0.5, hw_max=3.0, step=0.5)

        # Should have valid data
        self.assertIn('headwater', rating)
        self.assertIn('discharge', rating)
        self.assertTrue(len(rating['headwater']) > 0)

        # Discharge should increase with headwater
        for i in range(1, len(rating['discharge'])):
            self.assertGreaterEqual(rating['discharge'][i], rating['discharge'][i - 1])

    def test_downstream_channel_with_pipe_arch(self):
        """Test downstream channel with PipeArchCulvert."""
        channel = TrapezoidalChannel(
            depth=2.0,
            width=4.0,
            bank_inclination_m=2.0,
            inclination=0.001,
            roughness_n=0.035
        )

        culvert = PipeArchCulvert(
            span_width=1.8,
            rise_height=1.2,
            length=30,
            slope=0.01,
            headwater=1.5,
            downstream_channel=channel
        )

        Q, regime = culvert.controlling_flow()
        self.assertGreater(Q, 0)
        self.assertIn(regime, ['inlet', 'outlet'])

    def test_outlet_invert_offset_positive(self):
        """Test with outlet above channel bottom (positive offset)."""
        channel = RectangularChannel(
            depth=2.0,
            width=5.0,
            inclination=0.001,
            roughness_n=0.03
        )

        # Without offset
        culvert_no_offset = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=2.0,
            downstream_channel=channel, outlet_invert_offset=0.0
        )
        Q1, _ = culvert_no_offset.controlling_flow()
        tw1 = culvert_no_offset.tailwater

        # With positive offset (outlet 0.3m above channel bottom)
        culvert_offset = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=2.0,
            downstream_channel=channel, outlet_invert_offset=0.3
        )
        Q2, _ = culvert_offset.controlling_flow()
        tw2 = culvert_offset.tailwater

        # Tailwater should be lower when outlet is elevated
        self.assertLess(tw2, tw1)
        # Flow should be higher with lower tailwater
        self.assertGreaterEqual(Q2, Q1)

    def test_outlet_invert_offset_negative(self):
        """Test with outlet below channel bottom (negative offset)."""
        channel = RectangularChannel(
            depth=2.0,
            width=5.0,
            inclination=0.001,
            roughness_n=0.03
        )

        # Without offset
        culvert_no_offset = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=2.0,
            downstream_channel=channel, outlet_invert_offset=0.0
        )
        Q1, _ = culvert_no_offset.controlling_flow()
        tw1 = culvert_no_offset.tailwater

        # With negative offset (outlet 0.2m below channel bottom)
        culvert_offset = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=2.0,
            downstream_channel=channel, outlet_invert_offset=-0.2
        )
        Q2, _ = culvert_offset.controlling_flow()
        tw2 = culvert_offset.tailwater

        # Tailwater should be higher when outlet is lower
        self.assertGreater(tw2, tw1)

    def test_irregular_channel_nonzero_bottom(self):
        """Test with IrregularChannel where min elevation is not 0."""
        # Channel with bottom at elevation 100.0
        channel = IrregularChannel(
            stations=[0, 5, 10, 15, 20],
            elevations=[105, 102, 100, 102, 105],
            water_level=103.0,
            inclination=0.001,
            roughness_n=0.03
        )

        culvert = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=2.0,
            downstream_channel=channel
        )

        Q, regime = culvert.controlling_flow()
        self.assertGreater(Q, 0)
        # Tailwater should be water depth, not absolute elevation
        # Channel bottom = 100, so tailwater should be reasonable depth
        self.assertLess(culvert.tailwater, 10)  # Not an absolute elevation
        self.assertGreater(culvert.tailwater, 0)

    def test_irregular_channel_with_offset(self):
        """Test IrregularChannel with outlet_invert_offset."""
        # Channel with bottom at elevation 100.0
        channel = IrregularChannel(
            stations=[0, 5, 10, 15, 20],
            elevations=[105, 102, 100, 102, 105],
            water_level=103.0,
            inclination=0.001,
            roughness_n=0.03
        )

        # Outlet at elevation 100.5 (0.5m above channel bottom at 100.0)
        culvert = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=2.0,
            downstream_channel=channel,
            outlet_invert_offset=0.5
        )

        Q, _ = culvert.controlling_flow()
        self.assertGreater(Q, 0)

        # Compare with no offset
        culvert_no_offset = CircularCulvert(
            diameter=1.2, length=30, slope=0.01, headwater=2.0,
            downstream_channel=channel,
            outlet_invert_offset=0.0
        )
        Q_no_offset, _ = culvert_no_offset.controlling_flow()

        # With outlet elevated, tailwater is lower, so flow should be >=
        self.assertLess(culvert.tailwater, culvert_no_offset.tailwater)

    def test_tailwater_cannot_be_negative(self):
        """Tailwater should never be negative even with large positive offset."""
        channel = RectangularChannel(
            depth=1.0,
            width=3.0,
            inclination=0.001,
            roughness_n=0.03
        )

        # Large offset - outlet way above channel water
        culvert = CircularCulvert(
            diameter=1.0, length=20, slope=0.01, headwater=1.5,
            downstream_channel=channel,
            outlet_invert_offset=2.0  # Outlet 2m above channel bottom
        )

        Q, _ = culvert.controlling_flow()
        # Tailwater should be 0 (water doesn't reach outlet)
        self.assertEqual(culvert.tailwater, 0.0)


if __name__ == '__main__':
    unittest.main()
