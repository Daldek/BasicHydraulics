import math
from fractions import Fraction

'''
https://en.wikipedia.org/wiki/Manning_formula
'''

class Channel():
    '''Simple class to define channel's XS'''

    def __init__(self, inclination, roughness_n):
        '''Basic input parameters'''
        self.inclination = inclination
        self.roughness_n = roughness_n
        self._hydraulic_radius = None
        self._velocity = None
        self._flow_rate = None

    def hydraulic_radius(self):
        '''
        https://en.wikipedia.org/wiki/Manning_formula#Hydraulic_radius
        '''
        if self._hydraulic_radius is None:
            self._hydraulic_radius = self.cross_sectional_area()/self.wetted_perimeter()
        return self._hydraulic_radius

    def velocity(self):
        if self._velocity is None:
            self._velocity = (1/self.roughness_n) *\
            math.pow(self.hydraulic_radius(), Fraction(2, 3)) *\
            math.pow(self.inclination, 0.5)
        return self._velocity

    def flow_rate(self):
        if self._flow_rate is None:
            self._flow_rate = self.velocity()*self.cross_sectional_area()
        return self._flow_rate

class Flat(Channel):
    '''
    Class for flat surfaces
    '''
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

class TriangularChannel(Channel):
    '''Class for triangular channels'''

    def __init__(self, depth, bank_slope, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.bank_slope = bank_slope
        self._cross_sectional_area = None
        self._wetted_perimeter = None
        self._base_width = None
    
    def base_width(self):
        '''
        Calculated for half of the cross-section, i.e. a right-angled triangle.
        '''
        if self._base_width is None:
            self._base_width = self.depth * self.bank_slope
        print(f'Base width is {self._base_width} m')
        return self._base_width
    
    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = self.base_width() * self.depth
        print(f'XS area is {self._cross_sectional_area} sqm')
        return self._cross_sectional_area
    
    def wetted_perimeter(self):
        '''
        https://en.wikipedia.org/wiki/Wetted_perimeter
        '''
        if self._wetted_perimeter is None:
            self._wetted_perimeter = 2 * math.sqrt(math.pow(self.depth, 2) + math.pow(self.base_width(), 2))
        print(f'Wetter perimeter is {self._wetted_perimeter} m')
        return self._wetted_perimeter

class RectangularChannel(Channel):
    '''Class for rectangular channels'''

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
        '''
        https://en.wikipedia.org/wiki/Wetted_perimeter
        '''
        if self._wetted_perimeter is None:
            self._wetted_perimeter = 2 * self.depth + self.width
        return self._wetted_perimeter

    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, int((self.depth*100) + 1), 10):
            self.depth = h / 100.0
            self._cross_sectional_area = None
            self._wetted_perimeter = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            flow_rates[self.depth] = round(self.flow_rate(), 2)
        return flow_rates

class TrapezoidalChannel(Channel):
    '''Class for trapezoidal channels'''

    def __init__(self, depth, width, bank_inclination_m, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.width = width
        self.bank_inclination_m = bank_inclination_m
        self._cross_sectional_area = None
        self._wetted_perimeter = None

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = ((self.depth*self.bank_inclination_m)+self.width)*self.depth
        return self._cross_sectional_area

    def wetted_perimeter(self):
        '''
        https://en.wikipedia.org/wiki/Wetted_perimeter
        '''
        if self._wetted_perimeter is None:
            self._wetted_perimeter = self.width + 2 * self.depth *\
                  math.sqrt(1 + math.pow(self.bank_inclination_m, 2))
        return self._wetted_perimeter
    
    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, int((self.depth*100) + 1), 10):
            self.depth = h / 100.0
            self._cross_sectional_area = None
            self._wetted_perimeter = None
            self._hydraulic_radius = None
            self._velocity = None
            self._flow_rate = None
            flow_rates[self.depth] = round(self.flow_rate(), 2)
        return flow_rates

class SemiCircularChannel(Channel):
    '''Class for circular channels'''

    def __init__(self, depth, radius, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.radius = radius
        self._theta_rad = None
        self._cross_sectional_area = None
        self._wetted_perimeter = None

    def theta_rad(self):
        '''
        https://en.wikipedia.org/wiki/Radian
        https://en.wikipedia.org/wiki/Sine_and_cosine
        The theta angle was calculated as the cosine of the ratio of the height of the right triangle
        (radius of circle - depth of water (sagitta)), i.e. the perpendicular lying at the acute angle,
        to the radius, i.e. the opposite perpendicular.
        The result is then doubled to obtain the angle theta for an isosceles triangle
        '''
        if self._theta_rad is None:
            self._theta_rad = 2 * math.acos((self.radius - self.depth)/self.radius)
        return self._theta_rad

    def cross_sectional_area(self):
        '''
        https://en.wikipedia.org/wiki/Circular_segment
        '''
        if self._cross_sectional_area is None:
            self._cross_sectional_area = 0.5 * math.pow(self.radius, 2) * (self.theta_rad() - math.sin(self.theta_rad()))
        return self._cross_sectional_area

    def wetted_perimeter(self):
        '''
        https://en.wikipedia.org/wiki/Wetted_perimeter
        '''
        if self._wetted_perimeter is None:
            self._wetted_perimeter = self.radius * self.theta_rad()
        return self._wetted_perimeter
