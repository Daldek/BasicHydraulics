import math

g = 9.80665  # Standard gravity in m/s^2, see https://en.wikipedia.org/wiki/Standard_gravity


class Structure:
    """
    Base class representing a generic structure influenced by hydraulic head.

    Attributes:
        hydraulic_head (float): The hydraulic head or height of fluid column above a reference point.
        fluid_density (float): The density of the fluid in kg/m³ (default is water at 1000 kg/m³).
        dimension (float or list): The size of the opening, either as a single value for circular
                                   openings or a list of two values for rectangular openings.

    Methods:
        opening_area(): Calculate the area of the opening.
        get_opening_profile(): Return a dictionary describing the opening geometry.
    """

    def __init__(self, hydraulic_head=None, fluid_density=1000, dimension=None):
        """
        Initialize a Structure instance.

        Args:
            hydraulic_head (float): The hydraulic head of the structure.
            fluid_density (float, optional): The density of the fluid in kg/m³. Defaults to 1000 kg/m³ for water.
            dimension (float or list, optional): Dimension(s) of the opening, either as a single
                                                 radius for circular openings, or as a list of width and height.
        """
        self.hydraulic_head = hydraulic_head
        self.fluid_density = fluid_density
        self.dimension = dimension
        self._opening_area = None  # Cache for the opening area

    def opening_area(self):
        """
        Calculate and return the area of the opening.

        For a circular opening, it is calculated as πr^2/4. For rectangular openings, it is the product of width and height.

        Returns:
            float: The area of the opening.

        Raises:
            ValueError: If the dimension is not properly defined as a float/int or list of one or two values.
        """
        if self._opening_area is None:
            if isinstance(self.dimension, (int, float)):
                # Circular opening
                self._opening_area = math.pi * math.pow(self.dimension, 2) * 0.25
            elif isinstance(self.dimension, list) and len(self.dimension) == 1:
                # Circular opening (dimension as a list with one element)
                self._opening_area = math.pi * math.pow(self.dimension[0], 2) * 0.25
            elif isinstance(self.dimension, list) and len(self.dimension) == 2:
                # Rectangular opening
                self._opening_area = math.prod(self.dimension)
            else:
                raise ValueError(
                    "Dimension must be a single float/int or a list of one or two floats/ints."
                )
        return self._opening_area

    def get_opening_profile(self):
        """
        Return a dictionary describing the opening geometry.

        Returns:
            dict: Opening profile with type, dimensions, and area.
                  For circular: {'type': 'circular', 'diameter': float, 'area': float}
                  For rectangular: {'type': 'rectangular', 'width': float, 'height': float, 'area': float}
        """
        if isinstance(self.dimension, (int, float)):
            return {
                'type': 'circular',
                'diameter': self.dimension,
                'area': self.opening_area()
            }
        elif isinstance(self.dimension, list) and len(self.dimension) == 1:
            return {
                'type': 'circular',
                'diameter': self.dimension[0],
                'area': self.opening_area()
            }
        elif isinstance(self.dimension, list) and len(self.dimension) == 2:
            return {
                'type': 'rectangular',
                'width': self.dimension[0],
                'height': self.dimension[1],
                'area': self.opening_area()
            }
        else:
            raise ValueError(
                "Dimension must be a single float/int or a list of one or two floats/ints."
            )


class SmallOpening(Structure):
    """
    Class representing a small, non-submerged opening in a structure, such as a hole or orifice.
    Uses Torricelli's law for orifice flow calculations.

    Attributes:
        dimension (float or list): The size of the opening, either as a single value for circular
                                   openings or a list of two values for rectangular openings.
        hydraulic_head (float): The hydraulic head influencing the flow through the opening.
        velocity_coef (float): Coefficient representing the effect of velocity, typically 0.98.
        contraction_coef (float): Coefficient representing the effect of contraction, typically 0.64.

    Methods:
        pressure_at_opening(): Calculate pressure at the opening.
        velocity(): Calculate fluid velocity using Torricelli's law.
        flow(): Calculate flow rate through the opening.
        time_to_discharge(surface_area): Calculate time to discharge fluid.
        flow_duration_for_volume(volume): Calculate time to discharge specific volume.
        calculate_flow_curves(head_min, head_max, step): Generate flow rate curve Q(h).
    """

    def __init__(
        self,
        dimension=None,
        hydraulic_head=None,
        velocity_coef=0.98,
        contraction_coef=0.64,
        fluid_density=1000,
    ):
        """
        Initialize a SmallOpening instance.

        Args:
            dimension (float or list, optional): Dimension(s) of the opening, either as a single
                                                 radius for circular openings, or as a list of width and height.
            hydraulic_head (float, optional): The hydraulic head affecting the flow through the opening.
            velocity_coef (float, optional): Velocity coefficient. Defaults to 0.98.
            contraction_coef (float, optional): Contraction coefficient. Defaults to 0.64.
            fluid_density (float, optional): The density of the fluid in kg/m³. Defaults to 1000 kg/m³ (for water).
        """
        super().__init__(hydraulic_head, fluid_density, dimension)
        self.velocity_coef = velocity_coef
        self.contraction_coef = contraction_coef
        self._velocity = None
        self._flow = None

    def pressure_at_opening(self):
        """
        Calculate the pressure at the opening based on hydraulic head.

        Returns:
            float: Pressure in Pascals (Pa).
        """
        pressure = self.fluid_density * g * self.hydraulic_head
        return pressure

    def velocity(self):
        """
        Calculate and return the velocity of the fluid through the opening using Torricelli's law.
        The velocity is proportional to the square root of the hydraulic head and gravity.

        https://en.wikipedia.org/wiki/Torricelli%27s_law
        The velocity expressed by Torricelli's law is equal to the free fall velocity of
        a body from a height h. It is strictly valid for a non-viscous, incompressible liquid,
        if the diameter of the vessel is much larger than the diameter of the opening and
        neglecting changes in atmospheric pressure with a change in height by h and assuming
        that the upper surface of the liquid in the vessel is a free surface.

        Returns:
            float: The velocity of the fluid through the opening.
        """
        if self._velocity is None:
            self._velocity = self.velocity_coef * math.sqrt(2 * g * self.hydraulic_head)
        return self._velocity

    def flow(self):
        """
        Calculate and return the flow rate through the opening.
        The flow rate is the product of the contratcion coeficient, the opening area and the velocity of the fluid.

        Returns:
            float: The flow rate through the opening
        """
        if self._flow is None:
            self._flow = self.contraction_coef * self.opening_area() * self.velocity()
        return self._flow

    def time_to_discharge(self, surface_area):
        """
        Calculate the time required for the fluid to discharge through the opening without inflow.

        Uses the discharge coefficient (product of velocity coefficient and contraction coefficient)
        to estimate the discharge time, assuming free discharge and no inflow.
        https://en.wikipedia.org/wiki/Discharge_coefficient

        Args:
        surface_area (float, optional): The surface area of the fluid. Assumed as constant.

        Returns:
            float: The estimated time (s) for the fluid to fully discharge.
        """
        discharge_coef = self.velocity_coef * self.contraction_coef
        t = (
            (2 * surface_area)
            / (discharge_coef * self.opening_area() * math.sqrt(2 * g))
        ) * math.sqrt(self.hydraulic_head)
        return t

    def flow_duration_for_volume(self, volume):
        """
        Calculate the time needed to discharge a specific volume of fluid.

        Args:
            volume (float): Volume of fluid to discharge in cubic meters (m³).

        Returns:
            float: Time in seconds to discharge the specified volume.
        """
        time = volume / self.flow()
        return time

    def calculate_flow_curves(self, head_min=0.0, head_max=10.0, step=0.1):
        """
        Generate a mapping of hydraulic head to flow rate.

        Iterates through hydraulic head values from head_min to head_max,
        calculating the flow rate at each step. Compatible with the
        calculate_flow_rates() method in channel.py.

        Args:
            head_min (float): Minimum hydraulic head in meters. Defaults to 0.0.
            head_max (float): Maximum hydraulic head in meters. Defaults to 10.0.
            step (float): Increment step for hydraulic head. Defaults to 0.1.

        Returns:
            dict: Mapping of hydraulic head (float) to flow rate (float),
                  with values rounded to 4 decimal places.
        """
        original_head = self.hydraulic_head
        flow_curves = {}
        head = head_min

        while head <= head_max + step / 2:  # Add tolerance for floating point
            rounded_head = round(head, 2)
            if rounded_head == 0.0:
                flow_curves[0.0] = 0.0
            else:
                self.hydraulic_head = rounded_head
                self._velocity = None
                self._flow = None
                flow_curves[rounded_head] = round(self.flow(), 4)
            head += step

        self.hydraulic_head = original_head
        self._velocity = None
        self._flow = None

        return flow_curves


class LargeOpening(Structure):
    """
    Class representing steady flow through a large opening.
    Accounts for varying head across opening height using Boussinesq formula.

    Attributes:
        dimension (float or list): The size of the opening, either as a single value for circular
                                   openings or a list of two values for rectangular openings.
        hydraulic_head (float): The hydraulic head influencing the flow through the opening.
        fluid_density (float): The density of the fluid in kg/m³.
        discharge_coef (float): The discharge coefficient for the opening (typically between 0.6 and 1).

    Methods:
        free_flow(): Calculate flow rate for unsubmerged discharge.
        submerged_flow(height_difference): Calculate flow rate for fully submerged opening.
        partially_submerged_flow(h1, h2, h3): Calculate flow rate for mixed conditions.
        calculate_flow_curves(head_min, head_max, step, flow_type): Generate flow rate curves Q(h).
    """

    def __init__(
        self,
        dimension=None,
        hydraulic_head=None,
        discharge_coef=0.98,
        fluid_density=1000,
    ):
        """
        Initialize a LargeOpeningFreeFlow instance.

        Args:
            dimension (float or list): Dimension(s) of the opening, either as a single
                                       radius for circular openings, or as a list of width and height.
            hydraulic_head (float): The hydraulic head of the structure.
            discharge_coef (float, optional): The discharge coefficient. Defaults to 0.98.
            fluid_density (float): The density of the fluid in kg/m³. Defaults to 1000 kg/m³ for water.
        """
        super().__init__(hydraulic_head, fluid_density, dimension)
        self.discharge_coef = discharge_coef  # Coefficient of discharge

    def calculate_h(self, add=True):
        """
        Calculate either h1 (hydraulic_head + half the dimension) or h2 (hydraulic_head - half the dimension),
        depending on the value of the 'add' argument.

        Args:
            add (bool): If True, calculates h1 (addition). If False, calculates h2 (subtraction).

        Returns:
            float: Calculated h1 or h2 value.
        """
        if isinstance(self.dimension, (int, float)):
            half_dimension = self.dimension / 2.0
        elif isinstance(self.dimension, list) and len(self.dimension) == 1:
            half_dimension = self.dimension[0] / 2.0
        elif isinstance(self.dimension, list) and len(self.dimension) == 2:
            half_dimension = (
                self.dimension[1] / 2.0
            )  # For rectangular openings, use the second dimension
        else:
            raise ValueError(
                "Dimension must be a float/int or a list of one or two floats/ints."
            )

        return (
            self.hydraulic_head + half_dimension
            if add
            else self.hydraulic_head - half_dimension
        )

    def free_flow(self):
        """
        Calculate flow rate for free discharge through a large opening.

        Flow for large openings uses the calculated values of h1 (addition) and h2 (subtraction).

        Returns:
            float: Flow rate through the large opening.
        """
        area = self.opening_area()
        h1_val = self.calculate_h(add=True)
        h2_val = self.calculate_h(add=False)

        # Using a modified equation for large openings (Boussinesq formula or similar)
        flow_rate = (
            (2.0 / 3.0)
            * self.discharge_coef
            * area
            * math.sqrt(2 * g)
            * (h1_val ** (3 / 2) - h2_val ** (3 / 2))
        )
        return flow_rate

    def submerged_flow(self, height_difference):
        """
        Calculate flow rate for a submerged opening (completely underwater).

        The submerged flow formula is based on Torricelli's law, which calculates the velocity of fluid
        exiting the opening based on the difference in fluid heights inside and outside the structure.

        Args:
            height_difference (float): The difference in fluid levels inside and outside the structure.

        Returns:
            float: Flow rate for submerged discharge through the opening.
        """

        # Using Torricelli's law for submerged flow
        velocity = math.sqrt(
            2 * g * height_difference
        )  # Velocity of fluid based on submersion depth

        # The discharge flow rate includes the discharge coefficient and the velocity of the submerged flow
        flow_rate = self.discharge_coef * self.opening_area() * velocity
        return flow_rate

    def partially_submerged_flow(self, h1, h2, h3):
        """
        Calculate flow rate for a partially submerged opening.

        The flow is treated as the sum of the flow through the submerged part (h1)
        and the free flow through the non-submerged part (h2 and h3).

        Args:
            h1 (float): The distance from the water surface to the top of the opening (submerged part).
            h2 (float): The distance from the top of the opening to the water surface (non-submerged part).
            h3 (float): The distance from the water surface (non-submerged) to the bottom of the opening.

        Returns:
            float: Flow rate for the partially submerged discharge through the opening.
        """
        flow_rate = (
            (2 / 3)
            * self.discharge_coef
            * self.dimension[0]
            * math.sqrt(2 * g)
            * (math.pow((h1 + h2), (3.0 / 2.0)) - math.pow(h1, (3.0 / 2.0)))
        ) + (
            self.discharge_coef * self.dimension[0] * h3 * math.sqrt(2 * g * (h1 + h2))
        )
        return flow_rate

    def calculate_flow_curves(self, head_min=0.0, head_max=10.0, step=0.1, flow_type='free'):
        """
        Generate a mapping of hydraulic head to flow rate.

        Iterates through hydraulic head values from head_min to head_max,
        calculating the flow rate at each step. Compatible with the
        calculate_flow_rates() method in channel.py.

        Note: For free flow, head must be >= half the opening height to avoid
        negative values under the square root. Points below this threshold
        return 0.0.

        Args:
            head_min (float): Minimum hydraulic head in meters. Defaults to 0.0.
            head_max (float): Maximum hydraulic head in meters. Defaults to 10.0.
            step (float): Increment step for hydraulic head. Defaults to 0.1.
            flow_type (str): Type of flow calculation. Options:
                - 'free': Free discharge flow (default)
                - 'submerged': Submerged flow (uses head as height_difference)
                - 'all': Returns both flow types in nested dict

        Returns:
            dict: For 'free' or 'submerged': mapping of hydraulic head (float)
                  to flow rate (float), with values rounded to 4 decimal places.
                  For 'all': {'free': {...}, 'submerged': {...}}
        """
        original_head = self.hydraulic_head
        flow_curves = {}
        head = head_min

        # Determine minimum head for free flow (head must be >= half opening height)
        if isinstance(self.dimension, (int, float)):
            min_free_head = self.dimension / 2
        elif isinstance(self.dimension, list) and len(self.dimension) == 1:
            min_free_head = self.dimension[0] / 2
        elif isinstance(self.dimension, list) and len(self.dimension) == 2:
            min_free_head = self.dimension[1] / 2  # height
        else:
            min_free_head = 0

        if flow_type == 'all':
            free_curves = {}
            submerged_curves = {}
            while head <= head_max + step / 2:
                rounded_head = round(head, 2)
                if rounded_head == 0.0:
                    free_curves[0.0] = 0.0
                    submerged_curves[0.0] = 0.0
                else:
                    self.hydraulic_head = rounded_head
                    # Calculate free flow (0.0 if head below threshold)
                    if rounded_head >= min_free_head:
                        free_curves[rounded_head] = round(self.free_flow(), 4)
                    else:
                        free_curves[rounded_head] = 0.0
                    submerged_curves[rounded_head] = round(self.submerged_flow(rounded_head), 4)
                head += step
            self.hydraulic_head = original_head
            return {'free': free_curves, 'submerged': submerged_curves}

        while head <= head_max + step / 2:
            rounded_head = round(head, 2)
            if rounded_head == 0.0:
                flow_curves[0.0] = 0.0
            else:
                self.hydraulic_head = rounded_head
                if flow_type == 'submerged':
                    flow_curves[rounded_head] = round(self.submerged_flow(rounded_head), 4)
                else:  # 'free' or default
                    # Return 0.0 if head below threshold for free flow
                    if rounded_head >= min_free_head:
                        flow_curves[rounded_head] = round(self.free_flow(), 4)
                    else:
                        flow_curves[rounded_head] = 0.0
            head += step

        self.hydraulic_head = original_head
        return flow_curves


def plot_structure(structure, head_max=None, title=None):
    """
    Visualize flow rate curve for a structure.

    Args:
        structure: Structure object (SmallOpening or LargeOpening)
        head_max: Maximum hydraulic head for flow curve. Defaults to 2x current head.
        title: Optional title for the plot

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    hydraulic_head = structure.hydraulic_head or 1.0

    # Generate flow curve
    if head_max is None:
        head_max = max(hydraulic_head * 2, 1.0)

    flow_curves = structure.calculate_flow_curves(head_max=head_max, step=0.1)
    heads = list(flow_curves.keys())
    flows = list(flow_curves.values())

    sns.lineplot(x=flows, y=heads, ax=ax, marker='o', markersize=4)
    ax.set_xlabel('Flow rate [m³/s]')
    ax.set_ylabel('Hydraulic head [m]')

    # Mark current operating point
    if isinstance(structure, SmallOpening):
        current_flow = structure.flow()
    else:  # LargeOpening
        current_flow = structure.free_flow()

    ax.scatter([current_flow], [hydraulic_head], color='red', s=100, zorder=5)
    ax.annotate(f'Q={current_flow:.3f} m³/s',
                (current_flow, hydraulic_head),
                textcoords="offset points", xytext=(10, 10), color='red')

    ax.set_title(title or 'Flow rate curve')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig
