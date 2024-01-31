import math
from fractions import Fraction

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
        if self._hydraulic_radius is None:
            self._hydraulic_radius = self.cross_sectional_area()/self.wetted_circut()
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

class RectangularChannel(Channel):
    '''Class for rectangular channels'''

    def __init__(self, depth, width, inclination, roughness_n):
        super().__init__(inclination, roughness_n)
        self.depth = depth
        self.width = width
        self._cross_sectional_area = None
        self._wetted_circut = None

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = self.width * self.depth
        return self._cross_sectional_area

    def wetted_circut(self):
        if self._wetted_circut is None:
            self._wetted_circut = 2 * self.depth + self.width
        return self._wetted_circut

    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, (self.depth*100) + 1, 10):
            self.depth = h / 100.0
            self._cross_sectional_area = None
            self._wetted_circut = None
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
        self._wetted_circut = None

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = ((self.width+((2*(self.depth*self.bank_inclination_m))+self.width))/2.0)*self.depth
        return self._cross_sectional_area

    def wetted_circut(self):
        if self._wetted_circut is None:
            xx = math.pow((self.bank_inclination_m*self.depth), 2)+math.pow(self.depth, 2)
            x = math.sqrt(xx)
            self._wetted_circut = (2*x)+self.width
        return self._wetted_circut
    
    def calculate_flow_rates(self):
        flow_rates = {}
        for h in range(0, (self.depth*100) + 1, 10):
            self.depth = h / 100.0
            self._cross_sectional_area = None
            self._wetted_circut = None
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
        self._wetted_circut = None

    def theta_rad(self):
        if self._theta_rad is None:
            self._theta_rad = 2 * math.acos((self.radius - self.depth)/self.radius)
        return self._theta_rad

    def cross_sectional_area(self):
        if self._cross_sectional_area is None:
            self._cross_sectional_area = 0.5 * math.pow(self.radius, 2) * (self.theta_rad() - math.sin(self.theta_rad()))
        return self._cross_sectional_area

    def wetted_circut(self):
        if self._wetted_circut is None:
            self._wetted_circut = self.radius * self.theta_rad()
        return self._wetted_circut
