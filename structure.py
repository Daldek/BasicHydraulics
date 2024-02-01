import math

g = 9.80665  # https://en.wikipedia.org/wiki/Standard_gravity

class Structure():
    
    def __init__(self, hydraulic_head):
        self.hydraulic_head = hydraulic_head


class SmallOpening(Structure):
    '''
    Class for a small non-submerged opening
    '''

    def __init__(self, dimension, discharge_coef, hydraulic_head):
        super().__init__(hydraulic_head)
        self.dimension = dimension  # dimension or width and height
        self.discharge_coef = discharge_coef
        self._opening_area = None
        self._velocity = None
        self._flow = None

    def opening_area(self):
        if self._opening_area is None:
            if isinstance(self.dimension, (int, float)):
                # circular opening
                self._opening_area = math.pi * math.pow(self.dimension, 2) * 0.25
            elif len(self.dimension) == 1:
                # circular opening
                self._opening_area = math.prod(self.dimension)
            else:
                # square opening
                self._opening_area = math.prod(self.dimension)
        print(self._opening_area)
        return self._opening_area
    
    def velocity(self):
        '''
        https://en.wikipedia.org/wiki/Torricelli%27s_law
        The velocity expressed by Torricelli's law is equal to the free fall velocity of
        a body from a height h. It is strictly valid for a non-viscous, incompressible liquid,
        if the diameter of the vessel is much larger than the diameter of the opening and
        neglecting changes in atmospheric pressure with a change in height by h and assuming
        that the upper surface of the liquid in the vessel is a free surface.
        '''
        if self._velocity is None:
            self._velocity = math.sqrt(2 * g * self.hydraulic_head)
        print(self._velocity)
        return self._velocity
    
    def flow(self):
        '''
        https://en.wikipedia.org/wiki/Discharge_coefficient
        Discharge coefficient (also known as coefficient of discharge or efflux coefficient)
        is the ratio of the actual discharge to the ideal discharge
        '''
        if self._flow is None:
            self._flow = self.discharge_coef * self.opening_area() * self.velocity()
        print(self._flow)
        return self._flow
