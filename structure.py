import math

g = 9.80665  # https://en.wikipedia.org/wiki/Standard_gravity


class Structure:

    def __init__(self, hydraulic_head):
        self.hydraulic_head = hydraulic_head


class SmallOpening(Structure):
    """
    Class for a small non-submerged opening
    """

    def __init__(
        self,
        dimension=None,
        hydraulic_head=None,
        surface_area=None,
        velocity_coef=0.98,
        contraction_coef=0.64,
    ):
        super().__init__(hydraulic_head)
        self.dimension = dimension  # dimension or width and height
        self.surface_area = surface_area  # assumed as constant
        self.velocity_coef = velocity_coef
        self.contraction_coef = contraction_coef
        self._opening_area = None
        self._velocity = None
        self._flow = None
        """Discharge coef. for water is typicly 0,98."""

    def opening_area(self):
        if self._opening_area is None:
            if isinstance(self.dimension, (int, float)):
                # circular opening
                self._opening_area = math.pi * math.pow(self.dimension, 2) * 0.25
            elif isinstance(self.dimension, list) and len(self.dimension) == 1:
                # circular opening
                self._opening_area = math.pi * math.pow(self.dimension[0], 2) * 0.25
            elif isinstance(self.dimension, list) and len(self.dimension) == 2:
                # square opening
                self._opening_area = math.prod(self.dimension)
            else:
                raise ValueError(
                    "Dimension must be a single float or int\
or a list of one or two floats or ints."
                )
        return self._opening_area

    def velocity(self):
        """
        https://en.wikipedia.org/wiki/Torricelli%27s_law
        The velocity expressed by Torricelli's law is equal to the free fall velocity of
        a body from a height h. It is strictly valid for a non-viscous, incompressible liquid,
        if the diameter of the vessel is much larger than the diameter of the opening and
        neglecting changes in atmospheric pressure with a change in height by h and assuming
        that the upper surface of the liquid in the vessel is a free surface.
        """
        if self._velocity is None:
            self._velocity = self.velocity_coef * math.sqrt(2 * g * self.hydraulic_head)
        return self._velocity

    def flow(self):
        """
        https://en.wikipedia.org/wiki/Discharge_coefficient
        Discharge coefficient (also known as coefficient of discharge or efflux coefficient)
        is the ratio of the actual discharge to the ideal discharge
        """
        if self._flow is None:
            self._flow = self.opening_area() * self.velocity()
        return self._flow

    def time_to_discharge(self):
        discharge_coef = self.velocity_coef * self.contraction_coef
        t = (
            (2 * self.surface_area)
            / (discharge_coef * self.opening_area() * math.sqrt(2 * g))
        ) * math.sqrt(self.hydraulic_head)
        return t
